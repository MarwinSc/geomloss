from pathlib import Path
import moderngl
from render.base import OrbitDragCameraWindow
from pyrr import Matrix44
import meshio
import numpy as np
import imgui
import scene.file_import as file_import
from optimal_transport.__main__ import main as ot_main
from optimal_transport.__main__ import direct_run, naive_direct_run, direct_run_, barycenter_run, run_with_labels, otot
from tkinter.filedialog import askopenfilenames
import render.util as util
import subprocess
import json 
import os
import pathlib

from scene.ensemble import Ensemble

def load_shader(shader_path):
    """Reads the shader code from a file."""
    with open(shader_path, 'r') as file:
        return file.read()

class Renderer(OrbitDragCameraWindow):

    aspect_ratio = None
    gl_version = (4, 6)
    resource_dir = Path(__file__).parents[3].resolve()
    title = "Morph"
    loaded = False
    assignments = []
    points = []
    num_points = []

    ens = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # vertex and fragment shader

        vertex_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "vertex_shader.glsl")
        fragment_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "fragment_shader.glsl")

        self.prog = self.ctx.program(
            vertex_shader=vertex_shader_code,
            fragment_shader=fragment_shader_code
        )

        # Load compute shader
        self.WORKGOUP_SIZE = 256
        compute_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "compute_shader.glsl")
        compute_shader_code_parsed = compute_shader_code.replace("%COMPUTE_SIZE%", str(self.WORKGOUP_SIZE))
        self.compute_shader = self.ctx.compute_shader(compute_shader_code_parsed)

        # create quad shader 
        vertex_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "screen_quad_vertex.glsl")
        fragment_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "screen_quad_fragment.glsl")
        self.quad_prog = self.ctx.program(
            vertex_shader=vertex_shader_code,
            fragment_shader=fragment_shader_code
        )

        # create quad vao 
        quad_vertices = np.array([
            -1.0, -1.0, 0.0, 0.0,  # Bottom-left  (pos: -1, -1) (UV: 0, 0)
            1.0, -1.0, 1.0, 0.0,  # Bottom-right (pos:  1, -1) (UV: 1, 0)
            -1.0,  1.0, 0.0, 1.0,  # Top-left     (pos: -1,  1) (UV: 0, 1)
            1.0,  1.0, 1.0, 1.0,  # Top-right    (pos:  1,  1) (UV: 1, 1)
        ], dtype='f4')

        quad_indices = np.array([0, 1, 2, 2, 1, 3], dtype='i4')

        # Create buffers
        vbo = self.ctx.buffer(quad_vertices)
        ibo = self.ctx.buffer(quad_indices)
        
        self.quad_vao = self.ctx.vertex_array(self.quad_prog, [
            (vbo, "2f 2f", "aPos", "aTexCoords")
        ], ibo)

        # todo ? 
        #self.wnd.mouse_exclusivity = True

        self.camera.projection.update(near=0.1, far=1000.0)
        self.camera.mouse_sensitivity = 0.75
        self.camera.zoom_sensitivity = 0.33
        self.camera.zoom = 2.5

        # imgui variables 
        self.transition_state = 0.0
        self.color_state = 0.0
        self.color_distance = False
        self.varying_size = False
        self.lock_states = True
        ## Rendering
        self.point_size = 3.0
        self.bg_color = (1.0, 1.0, 1.0)
        self.depth_contour_color = (0.0, 0.0, 0.0)
        self.exen_contour_color = (0.0, 0.0, 0.0)
        self.wireframe = False
        self.offset_factor = 1.0
        self.transparency = 1.0
        ## contour
        self.contour_overlay = True
        self.render_depth_contour = False
        self.depth_contour_amp = 5.0
        self.render_exen_contour = False
        self.exen_contour_amp = 1.0
        ## OT
        self.normalize_data = False
        self.accumulate_distance = True
        self.ot_blur = 0.001
        self.ot_scaling = 0.7
        self.ot_trunctate = 5

    def render(self, time: float, frametime: float):

        self.my_framebuffer.use()
        #self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.clear(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0)
        #self.ctx.clear(0.5, 0.5, 0.5, 1.0)
        
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING

        if self.loaded:
            
            # load next model
            if self.transition_state > (self.current_assignment + 1) * (1/(self.number_of_files - 1)):

                # get the data
                source_pos, target_pos = self.ens.increment()
                # assign the old target as the new source
                self.source_buffer = self.ctx.buffer(source_pos)
                # assign the new target
                self.target_buffer = self.ctx.buffer(target_pos)

                #self.compute_buffer_a = self.ctx.buffer(source_pos)
                self.compute_buffer_b = self.ctx.buffer(source_pos)

                #self.points_a = self.ctx.vertex_array(
                #    self.prog, [self.compute_buffer_a.bind('in_position', 'in_color', layout='4f 4f')],
                #)
                self.points_b = self.ctx.vertex_array(
                    self.prog, [self.compute_buffer_b.bind('in_position', 'in_color', layout='4f 4f')],
                )

                self.current_assignment += 1

            # load previous model
            elif self.transition_state < (self.current_assignment) * (1/(self.number_of_files - 1)):

                # get the data
                source_pos, target_pos = self.ens.decrement()
                # todo from here on  but the last line, a copy from above
                # assign the old target as the new source
                self.source_buffer = self.ctx.buffer(source_pos)
                # assign the new target
                self.target_buffer = self.ctx.buffer(target_pos)

                #self.compute_buffer_a = self.ctx.buffer(source_pos)
                self.compute_buffer_b = self.ctx.buffer(source_pos)

                #self.points_a = self.ctx.vertex_array(
                #    self.prog, [self.compute_buffer_a.bind('in_position', 'in_color', layout='4f 4f')],
                #)
                self.points_b = self.ctx.vertex_array(
                    self.prog, [self.compute_buffer_b.bind('in_position', 'in_color', layout='4f 4f')],
                )

                self.current_assignment -= 1


            if self.lock_states:
                if self.transition_state != self.color_state:
                    self.color_state = self.transition_state

            # Bind the appropriate buffers to the compute shader
            self.source_buffer.bind_to_storage_buffer(0)
            self.compute_buffer_b.bind_to_storage_buffer(1)
            self.target_buffer.bind_to_storage_buffer(2)

            #try:
            #    self.compute_shader['time'] = time
            #except Exception as e:
            #    #pass
            #    #TODO
            #    print(f"exception: {e}")
            #self.compute_shader['max_distance'] = self.max_distance
            self.compute_shader['transition_state'] = self.transition_state * (self.number_of_files - 1) - self.current_assignment
            self.compute_shader['color_state'] = self.color_state * (self.number_of_files - 1) - self.current_assignment
            # always take the number of points from the reference model
            self.compute_shader.run(group_x = int(np.ceil(self.num_points[0] / self.WORKGOUP_SIZE)))

            self.prog['projection'].write(self.camera.projection.matrix)
            self.prog['modelview'].write(self.camera.matrix)

            self.prog['point_size'] = self.point_size
            #self.prog['time'].value = time
            self.prog['varying_size'] = self.varying_size
            self.prog['transparency'] = self.transparency
            self.points_b.render(mode=self.ctx.POINTS)

        # bind the default framebuffer
        self.ctx.screen.use()
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.quad_prog['color_distance'] = self.color_distance
        self.quad_prog['colorTexture'].value = 0
        self.quad_prog['explicitEncodingTexture'].value = 1
        self.quad_prog['depthTexture'].value = 2
        self.my_framebuffer.color_attachments[0].use(location=0)
        self.my_framebuffer.color_attachments[1].use(location=1)
        self.my_framebuffer.depth_attachment.use(location=2)
        self.quad_prog['bg_color'] = self.bg_color
        # contour uniforms
        self.quad_prog['contour_overlay'] = self.contour_overlay
        self.quad_prog['render_depth_contour'] = self.render_depth_contour
        self.quad_prog['depth_contour_amp'] = self.depth_contour_amp
        self.quad_prog['render_exen_contour'] = self.render_exen_contour
        self.quad_prog['exen_contour_amp'] = self.exen_contour_amp
        self.quad_prog['depth_contour_color'] = self.depth_contour_color
        self.quad_prog['exen_contour_color'] = self.exen_contour_color
        self.quad_prog['offset_h'] = 1.0 / (self.wnd.size[0] * self.offset_factor)
        self.quad_prog['offset_v'] = 1.0 / (self.wnd.size[1] * self.offset_factor)

        self.ctx.wireframe = self.wireframe

        self.quad_vao.render(moderngl.TRIANGLES)

        self.ctx.wireframe = False

            # not sure how moderngl handels buffer swapping
            # but removing this gives about 20fps more
            # and this also results in a flickering
            # switch buffers for rendering 
            #self.compute_buffer_a, self.compute_buffer_b = self.compute_buffer_b, self.compute_buffer_a
            #self.points_a, self.points_b = self.points_b, self.points_a

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
            self.files = list(askopenfilenames(filetypes = [('', '*e57'), ('', '*ply'), ('', '*.obj'), ('', '*.laz')]))

        if hasattr(self, "files"):
            if imgui.tree_node("Files", imgui.TREE_NODE_DEFAULT_OPEN):

                #for i, file in enumerate(self.files):
                #    #imgui.text(file)
                #    pressed, state = imgui.selectable(file.split("/")[-1], self.selected == i)
                #    if pressed:
                #        self.selected = i
                #        print(f"Selected: {file}")

                for i, file in enumerate(self.files):
                    pressed, state = imgui.selectable(file.split("/")[-1])
                    if imgui.is_item_active() and not imgui.is_item_hovered():
                        next = i + (-1 if imgui.get_mouse_drag_delta(0)[1] < 0 else 1)
                        if next >= 0 and next < len(self.files):
                            self.files[i], self.files[next] = self.files[next], self.files[i]
                            imgui.reset_mouse_drag_delta()

                imgui.tree_pop()

                _, self.normalize_data = imgui.checkbox("Normalize Data", self.normalize_data)
                _, self.accumulate_distance = imgui.checkbox("Accumulate Distance", self.accumulate_distance)

        run_assign = imgui.button("Build Correspondence")
        if run_assign:
            util.write_filelist_json(self.files)
            self.run_ot()
            #self.load_data()

        load_debug = imgui.button("Load DEBUG")
        if load_debug:
            self.run_debug()

        renderer_ui, _ = imgui.collapsing_header("Renderer", True)
        if renderer_ui:
            _, self.point_size = imgui.slider_float("P", self.point_size, 1.0, 30.0)
            _, self.bg_color = imgui.color_edit3("Background Color", *self.bg_color)
            _, self.wireframe = imgui.checkbox("Wireframe", self.wireframe)
            _, self.transparency = imgui.slider_float("Transparency", self.transparency, 0.0, 1.0)
            # contour 
            _, self.contour_overlay = imgui.checkbox("Ex Contour Overlay", self.contour_overlay)
            _, self.render_depth_contour = imgui.checkbox("Render Depth Contour", self.render_depth_contour)
            _, self.render_exen_contour = imgui.checkbox("Render Ex Contour", self.render_exen_contour)

            contour_params_ui, _ = imgui.collapsing_header("Contour Parameters", True)
            if contour_params_ui:
                _, self.depth_contour_amp = imgui.slider_float("CD", self.depth_contour_amp, 0.0, 100.0)
                _, self.exen_contour_amp = imgui.slider_float("CEX", self.exen_contour_amp, 0.0, 100.0)
                _, self.depth_contour_color = imgui.color_edit3("Depth Color", *self.depth_contour_color)
                _, self.exen_contour_color = imgui.color_edit3("Explicit Encoding Color", *self.exen_contour_color)
                _, self.offset_factor = imgui.slider_float("Offset Factor", self.offset_factor, 0.1, 2.0)

        optimal_transport, _ = imgui.collapsing_header("Optimal Transport", True)
        if optimal_transport:
            _, self.ot_blur = imgui.input_float("Blur", self.ot_blur)
            _, self.ot_scaling = imgui.input_float("Scaling", self.ot_scaling)
            _, self.ot_trunctate = imgui.input_float("Truncate", self.ot_trunctate)

        comparison, _ = imgui.collapsing_header("Comparison", True)
        if comparison:
            _, self.color_distance = imgui.checkbox("Color Distance", self.color_distance)
            _, self.varying_size = imgui.checkbox("Varying Size", self.varying_size)

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
        self.number_of_files = len(filelist['files'])

        conf = {
            "octree_node_size": 1000,
            "normalize_data": self.normalize_data,
            "autograd": True,
            "sort_emd": False,
            "accumulate_distance": self.accumulate_distance
        }

        self.ens = Ensemble(filelist, conf)
        self.ens.build()
        #self.ens.ot_sequential()
        conf = {
            "blur" : self.ot_blur,
            "scaling" : self.ot_scaling,
            "truncate" : self.ot_trunctate
        } 
        self.ens.ot_reference(conf)
        source_pos, target_pos = self.ens.compute_data

        # Create the two buffers the compute shader will write and read from
        self.current_assignment = 0
        #self.compute_buffer_a = self.ctx.buffer(source_pos)
        self.compute_buffer_b = self.ctx.buffer(source_pos)
        self.source_buffer = self.ctx.buffer(source_pos)
        self.target_buffer = self.ctx.buffer(target_pos)

        # Prepare vertex arrays to drawing points using the compute shader buffers are input
        # We use 4x4 (padding format)
        #self.points_a = self.ctx.vertex_array(
        #    self.prog, [self.compute_buffer_a.bind('in_position', 'in_color', layout='4f 4f')],
        #)
        self.points_b = self.ctx.vertex_array(
            self.prog, [self.compute_buffer_b.bind('in_position', 'in_color', layout='4f 4f')],
        )

        self.num_points = self.ens.get_num_points()
        self.loaded = True

if __name__ == '__main__':
    Renderer.run()