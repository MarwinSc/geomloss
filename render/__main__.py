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

                self.ens.increment()

                source_pos, target_pos = self.ens.compute_data

                self.compute_buffer_a = self.ctx.buffer(source_pos)
                self.assignment_buffer = self.ctx.buffer(target_pos)

                self.current_assignment += 1

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
            #self.compute_shader['max_distance'] = self.max_distance
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
        self.number_of_files = len(filelist['files'])

        self.ens = Ensemble(filelist)
        self.ens.build()
        self.ens.ot_sequential()
        source_pos, target_pos = self.ens.compute_data

        # Create the two buffers the compute shader will write and read from
        self.current_assignment = 0
        self.compute_buffer_a = self.ctx.buffer(source_pos)
        self.compute_buffer_b = self.ctx.buffer(source_pos)
        self.assignment_buffer = self.ctx.buffer(target_pos)

        # Prepare vertex arrays to drawing points using the compute shader buffers are input
        # We use 4x4 (padding format)
        self.points_a = self.ctx.vertex_array(
            self.prog, [self.compute_buffer_a.bind('in_position', 'in_color', layout='4f 4f')],
        )
        self.points_b = self.ctx.vertex_array(
            self.prog, [self.compute_buffer_b.bind('in_position', 'in_color', layout='4f 4f')],
        )

        self.num_points = self.ens.get_num_points()
        self.loaded = True

if __name__ == '__main__':
    Renderer.run()