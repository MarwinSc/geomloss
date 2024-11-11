from pathlib import Path
import moderngl
from render.base import OrbitDragCameraWindow
from pyrr import Matrix44
import meshio
import numpy as np
import imgui
import util.file_import as file_import
from optimal_transport.__main__ import main as ot_main
from optimal_transport.__main__ import direct_run, naive_direct_run, direct_run_, barycenter_run, run_with_labels, otot
from tkinter.filedialog import askopenfilenames
import render.util as util
import subprocess
import json 
import os
import pathlib

compute_shader_code = """
    #version 460
    #define GROUP_SIZE %COMPUTE_SIZE%

    layout(local_size_x=GROUP_SIZE) in;

    //uniform float time;
    uniform float transition_state;
    uniform float color_state;
    uniform float max_distance;
    uniform bool color_distance;

    struct Point{
        vec4 pos;
        vec4 col;
    };

    struct Assignment{
        vec4 pos;  // w is the distance
        vec4 col;
    };

    layout(std430, binding=0) buffer points_in{
        Point points[];
    } In;
    layout(std430, binding=1) buffer points_out{
        Point points[];
    } Out;
    layout(std430, binding=2) buffer assignment{
        Assignment assignments[];
    } Ass;

    void main()
    {
        int x = int(gl_GlobalInvocationID);
        if(In.points.length() <= x){
            return;
        }

        Point in_point = In.points[x];
        vec4 p = in_point.pos.xyzw;

        vec3 target_p = Ass.assignments[x].pos.xyz;
        p.xyz = p.xyz * transition_state + target_p.xyz * (1 - transition_state);

        Point out_point;
        out_point.pos.xyz = p.xyz;
        out_point.pos.w = in_point.pos.w;

        if(color_distance){
            float d = distance(p.xyz, target_p.xyz);
            float interp = ((1 - (d / Ass.assignments[x].pos.w)) * (Ass.assignments[x].pos.w / max_distance));

            //vec4 c = in_point.col.xyzw;
            out_point.col.xyzw = vec4(0.0, 0.0, 1.0, 1.0) * (1 - interp) + vec4(1.0, 0.0, 0.0, 1.0) * interp;
        }else{
            out_point.col.xyzw = in_point.col.xyzw * color_state + Ass.assignments[x].col.xyzw * (1 - color_state);
            //out_point.col.xyzw = in_point.col.xyzw;
        }
        Out.points[x] = out_point;
    }
    """

vertex_shader_code = """
    #version 460

    in vec4 in_position;
    in vec4 in_color;

    uniform mat4 projection;
    uniform mat4 modelview;
    uniform float point_size;
    //uniform float time;

    out vec4 color;

    void main() {
        gl_Position = projection * modelview * in_position;
        vec4 position_camera_coord = modelview * in_position;
        // Set the point size
        //gl_PointSize = 25 - gl_Position.z + sin((time + gl_VertexID) * 7.0) * 10.0;
        gl_PointSize = point_size;

        // Calculate a random color based on the vertex index
        //color = vec3(mod(gl_VertexID * 432.43, 1.0), mod(gl_VertexID * 6654.32, 1.0), mod(gl_VertexID  * 6544.11, 1.0));
        color = in_color;
    }
    """

fragment_shader_code = """
    #version 460

    in vec4 color;
    out vec4 outColor;

    void main() {
        // Calculate the distance from the center of the point
        // gl_PointCoord is available when redering points. It's basically an uv coordinate.
        //float dist = step(length(gl_PointCoord.xy - vec2(0.5)), 0.5);

        // .. an use to render a circle!
        //outColor = vec4(dist * color, dist);

        outColor = color;
    }
    """

class Renderer(OrbitDragCameraWindow):
    """
    Example showing how to use a OrbitCamera
    """
    aspect_ratio = None
    gl_version = (4, 6)
    resource_dir = Path(__file__).parents[3].resolve()
    title = "Morph"
    loaded = False
    assignments = []
    points = []
    num_points = []

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.WORKGOUP_SIZE = 128

        self.prog = self.ctx.program(
            vertex_shader=vertex_shader_code,
            fragment_shader=fragment_shader_code
        )

        # Load compute shader
        compute_shader_code_parsed = compute_shader_code.replace("%COMPUTE_SIZE%", str(self.WORKGOUP_SIZE))
        self.compute_shader = self.ctx.compute_shader(compute_shader_code_parsed)

        # todo ? 
        #self.wnd.mouse_exclusivity = True

        self.camera.projection.update(near=0.1, far=1000.0)
        self.camera.mouse_sensitivity = 0.75
        self.camera.zoom = 2.5

        # imgui variables 
        self.transition_state = 0.0
        self.color_state = 0.0
        self.color_distance = False
        self.lock_states = True
        self.point_size = 3.0


    def render(self, time: float, frametime: float):
        #self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING

        if self.loaded:
            
            # load next model
            if self.transition_state > (self.current_assignment + 1) * (1/(self.number_of_files - 1)):
                self.current_assignment += 1

                # todo time this
                self.compute_buffer_a = self.ctx.buffer(self.points[self.current_assignment + 1])
                self.assignment_buffer = self.ctx.buffer(self.assignments[self.current_assignment])

                self.compute_buffer_a, self.compute_buffer_b = self.compute_buffer_b, self.compute_buffer_a
                self.points_a, self.points_b = self.points_b, self.points_a

            if self.lock_states:
                if self.transition_state != self.color_state:
                    self.color_state = self.transition_state

            # Calculate the next position of the balls with compute shader
            self.compute_buffer_a.bind_to_storage_buffer(0)
            self.compute_buffer_b.bind_to_storage_buffer(1)
            self.assignment_buffer.bind_to_storage_buffer(2)

            #try:
            #    self.compute_shader['time'] = time
            #except Exception as e:
            #    #pass
            #    #TODO
            #    print(f"exception: {e}")
            self.compute_shader['max_distance'] = self.max_distance
            self.compute_shader['color_distance'] = self.color_distance
            self.compute_shader['transition_state'] = self.transition_state * (self.number_of_files - 1) - self.current_assignment
            self.compute_shader['color_state'] = self.color_state * (self.number_of_files - 1) - self.current_assignment
            self.compute_shader.run(group_x = int(self.num_points[self.current_assignment + 1] / self.WORKGOUP_SIZE))

            self.prog['projection'].write(self.camera.projection.matrix)
            self.prog['modelview'].write(self.camera.matrix)

            self.prog['point_size'] = self.point_size
            #self.prog['time'].value = time
            self.points_b.render(mode=self.ctx.POINTS)

        self.render_ui()
    
    def render_ui(self):
        super().render_ui()
        imgui.new_frame()
        if imgui.begin_main_menu_bar():
            if imgui.begin_menu("File", True):

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True
                )

                clicked_quit, selected_quit = imgui.menu_item(
                    "Quit", 'Cmd+Q', False, True
                )

                if clicked_quit:
                    exit(1)

                imgui.end_menu()
            imgui.end_main_menu_bar()

        imgui.show_test_window()

        imgui.begin("Custom window", True)

        add_files = imgui.button("Files")
        if add_files:
            self.files = askopenfilenames(filetypes = [('', '*e57'), ('', '*ply'), ('', '*.obj'), ('', '*.laz')])
            util.write_filelist_json(self.files)
        if hasattr(self, "files"):
            for file in self.files:
                imgui.text(file)

        run_assign = imgui.button("Build Correspondence")
        if run_assign:
            self.run_ot()
            #self.load_data()

        load_debug = imgui.button("Load DEBUG")
        if load_debug:
            self.run_debug()

        _, self.color_distance = imgui.checkbox("Color Distance", self.color_distance)

        _, self.point_size = imgui.slider_float("", self.point_size, 1.0, 30.0)


        imgui.end()

        ##### slider

        wnd_size = self.wnd.size
        imgui.set_next_window_size(wnd_size[0], 100)
        imgui.set_next_window_position(0, wnd_size[1]-100)
        imgui.begin("State", False, flags=imgui.WINDOW_NO_COLLAPSE)
        _, self.lock_states = imgui.checkbox("Lock", self.lock_states)
        imgui.set_next_item_width(wnd_size[0] - (wnd_size[0] * 0.02))
        _, self.transition_state = imgui.slider_float("Transition", self.transition_state, 0.0, 1.0)
        imgui.set_next_item_width(wnd_size[0] - (wnd_size[0] * 0.02))
        _, self.color_state = imgui.slider_float("Color", self.color_state, 0.0, 1.0)
        imgui.end()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def run_debug(self):
        filelist = {"files": [
                                pathlib.Path(__file__).parents[3] / "data/statues/greif.ply",
                                pathlib.Path(__file__).parents[3] / "data/statues/loewe.ply"
                             ]         
                    }
        self.generic_run(filelist)

    def run_ot(self):
        filelist_path = util.create_tmp_dir() / "filelist.json"
        if filelist_path.is_file:
            with open(filelist_path, 'r') as infile:
                filelist = json.load(infile)
                self.generic_run(filelist)
                
    
    def generic_run(self, filelist):
        octrees = []
        mean = None
        for file in filelist['files']:
            if mean is None:
                octree, mean = file_import.read(file, None)
            else:
                octree, _ = file_import.read(file, mean)

            octrees.append(octree)

        correspondences, matching_colors = otot(octrees)
        #correspondences = naive_direct_run(octrees)
        #correspondences, matching_colors = direct_run_(octrees)
        #correspondences = barycenter_run(octrees)
        #correspondences = run_with_labels(octrees)
        
        self.number_of_files = len(filelist['files'])

        # perpare compute data e.g. positions and colors
        for i, oct in enumerate(octrees):

            positions = oct.points
            colors = oct.colors

            self.num_points.append(positions.shape[0])

            # swap columns due to blender
            positions[:,[1, 2]] = positions[:,[2, 1]]
            positions = np.c_[positions, np.ones(positions.shape[0])]

            compute_data = np.empty((positions.shape[0] + colors.shape[0], 4), dtype="f4")
            compute_data[0::2,:] = positions
            compute_data[1::2,:] = colors

            self.points.append(compute_data)

        # prepare transition data e.g. assigned positions and colors
        for i, corres in enumerate(correspondences):
            assignment_positions = corres[:]
            # swap columns due to blender
            assignment_positions[:,[1, 2]] = assignment_positions[:,[2, 1]]
            # todo 
            #positions = octrees[i].points
            #positions[:,[1, 2]] = positions[:,[2, 1]]
            assignment_distances = np.linalg.norm(assignment_positions - positions[:, :3], axis=1)
            # todo
            self.max_distance = np.max(assignment_distances)
            assignment = np.empty((len(assignment_positions) * 2, 4), dtype="f4")
            assignment[0::2,:] = np.c_[assignment_positions, assignment_distances]
            assignment[1::2,:] = matching_colors[i]
            assignment = assignment.astype("f4")
            self.assignments.append(assignment)

        # Create the two buffers the compute shader will write and read from
        self.current_assignment = 0
        self.compute_buffer_a = self.ctx.buffer(self.points[self.current_assignment + 1])
        self.compute_buffer_b = self.ctx.buffer(self.points[self.current_assignment + 1])
        self.assignment_buffer = self.ctx.buffer(self.assignments[self.current_assignment])

        # Prepare vertex arrays to drawing points using the compute shader buffers are input
        # We use 4x4 (padding format) to skip the velocity data (not needed for drawing the balls)
        self.points_a = self.ctx.vertex_array(
            self.prog, [self.compute_buffer_a.bind('in_position', 'in_color', layout='4f 4f')],
        )
        self.points_b = self.ctx.vertex_array(
            self.prog, [self.compute_buffer_b.bind('in_position', 'in_color', layout='4f 4f')],
        )

        self.loaded = True


if __name__ == '__main__':
    Renderer.run()