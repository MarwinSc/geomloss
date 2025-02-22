from pathlib import Path
import imgui.integrations
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

        # create depth edge shader 
        vertex_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "quad_vertex.glsl")
        fragment_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "depth_edge_fragment.glsl")
        self.depth_edge_prog = self.ctx.program(
            vertex_shader=vertex_shader_code,
            fragment_shader=fragment_shader_code
        )

        # create exen edge shader 
        vertex_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "quad_vertex.glsl")
        fragment_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "exen_edge_fragment.glsl")
        self.exen_edge_prog = self.ctx.program(
            vertex_shader=vertex_shader_code,
            fragment_shader=fragment_shader_code
        )

        # create blur shader 
        vertex_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "quad_vertex.glsl")
        fragment_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "gaussian_fragment.glsl")
        self.gaussian_prog = self.ctx.program(
            vertex_shader=vertex_shader_code,
            fragment_shader=fragment_shader_code
        )

        # create dilation shader 
        vertex_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "quad_vertex.glsl")
        fragment_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "dilation_fragment.glsl")
        self.dilation_prog = self.ctx.program(
            vertex_shader=vertex_shader_code,
            fragment_shader=fragment_shader_code
        )

        # create composite shader 
        vertex_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "quad_vertex.glsl")
        fragment_shader_code = load_shader(pathlib.Path(__file__).parents[1] / "shaders" / "composite_fragment.glsl")
        self.composite_prog = self.ctx.program(
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

        self.depth_edges_vao = self.ctx.vertex_array(self.depth_edge_prog, [
            (vbo, "2f 2f", "aPos", "aTexCoords")
        ], ibo)

        self.exen_edges_vao = self.ctx.vertex_array(self.exen_edge_prog, [
            (vbo, "2f 2f", "aPos", "aTexCoords")
        ], ibo)

        self.gaussia_vao = self.ctx.vertex_array(self.gaussian_prog, [
            (vbo, "2f 2f", "aPos", "aTexCoords")
        ], ibo)

        self.dilation_vao = self.ctx.vertex_array(self.dilation_prog, [
            (vbo, "2f 2f", "aPos", "aTexCoords")
        ], ibo)

        self.composite_vao = self.ctx.vertex_array(self.composite_prog, [
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
        self.lock_states = True
        ## Rendering
        self.point_size = 6.0
        self.varying_size = True
        self.bg_color = (1.0, 1.0, 1.0)
        self.wireframe = False
        self.offset_factor = 1.0
        self.transparency = 1.0
        self.fps = 0.0
        self.frame_count = 0
        self.last_time = 0.0
        self.current_time = 0.0
        self.easyease = True
        self.play = False
        self.play_speed = 0.1
        self.playback_direction = 1
        ## Comparison
        self.color_distance = False
        self.filter_treshold = 0.0
        ## contour
        self.contour_overlay = True
        self.depth_contour_color = (0.0, 0.0, 0.0)
        self.exen_contour_color = (0.0, 0.0, 0.0)
        # depth contour
        self.render_depth_contour = False
        self.depth_contour_amp = 5.0
        self.depth_dilation_iterations = 1
        self.depth_blur = False
        self.depth_sigma = 1.0
        self.depth_opaque = False
        # exen contour
        self.render_exen_contour = False
        self.exen_contour_amp = 1.5
        self.exen_dilation_iterations = 0
        self.exen_number_contour_lines = 5.0
        self.exen_opaque = True
        ## OT
        self.normalize_data = False
        #self.accumulate_distance = True
        self.ot_blur = 0.001
        self.ot_scaling = 0.7
        self.ot_trunctate = 5
        self.ot_reach = 10.0
        self.uniform_reference = False
        self.reference_n = 300000

        self.font = self.imgui.io.fonts.add_font_from_file_ttf(str(pathlib.Path(__file__).parents[1] / "util" / "FontAwesome.ttf"), 16)
        self.imgui.refresh_font_texture()
        imgui.get_io().font_global_scale = 1.3

    def render(self, time: float, frametime: float):

        self.frame_count += 1
        if (time - self.last_time) > 1.0:
            self.fps = self.frame_count / (time - self.last_time)
            self.last_time = time
            self.frame_count = 0

        if self.play:
            self.transition_state += self.play_speed * frametime * self.playback_direction
            if self.transition_state >= 1.0:
                self.transition_state = 1.0
                self.playback_direction = -self.playback_direction
            if self.transition_state <= 0.0:
                self.transition_state = 0.0
                self.playback_direction = -self.playback_direction

        if self.lock_states:
            if self.transition_state != self.color_state:
                self.color_state = self.transition_state

        self.current_fbo, self.back_fbo = self.fbo_1, self.fbo_2

        self.my_framebuffer.use()
        #self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.PROGRAM_POINT_SIZE | moderngl.BLEND)
        self.ctx.clear(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0)
        #self.ctx.clear(0.5, 0.5, 0.5, 1.0)
        
        self.ctx.enable_only(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = moderngl.DEFAULT_BLENDING
        self.ctx.blend_equation = moderngl.FUNC_ADD
        
        if self.loaded:

            # TODO missing colors here.
            
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
            self.compute_shader['transition_state'] = self.get_transition_state()
            self.compute_shader['color_state'] = self.get_color_state()
            # always take the number of points from the reference model
            self.compute_shader.run(group_x = int(np.ceil(self.num_points[0] / self.WORKGOUP_SIZE)))

            self.prog['projection'].write(self.camera.projection.matrix)
            self.prog['modelview'].write(self.camera.matrix)

            self.prog['point_size'] = self.point_size
            #self.prog['time'].value = time
            self.prog['color_state'] = self.get_color_state()
            self.prog['varying_size'] = self.varying_size
            self.prog['transparency'] = self.transparency
            self.prog['filter_treshold'] = self.filter_treshold

            self.points_b.render(mode=self.ctx.POINTS)

        self.ctx.disable(moderngl.BLEND)

        if self.render_depth_contour:
            self.render_depth_edges()
        else:
            self.depth_edges_fbo.clear(0.0, 0.0, 0.0, 0.0)

        if self.render_exen_contour:
            self.render_exen_edges()
        else:
            self.exen_edges_fbo.clear(0.0, 0.0, 0.0, 0.0)

        self.ctx.enable(moderngl.BLEND)

        self.ctx.screen.use()
        self.ctx.clear(self.bg_color[0], self.bg_color[1], self.bg_color[2], 1.0)

        self.composite_prog['contour_overlay'] = self.contour_overlay
        self.composite_prog['colorTexture'].value = 0
        self.composite_prog['depthEdgesTexture'].value = 1
        self.composite_prog['exenEdgesTexture'].value = 2

        if self.color_distance:
            self.my_framebuffer.color_attachments[1].use(location=0)
        else:
            self.my_framebuffer.color_attachments[0].use(location=0)
        # todo change when using blur
        self.depth_edges_fbo.color_attachments[0].use(location=1)
        self.exen_edges_fbo.color_attachments[0].use(location=2)

        self.ctx.wireframe = self.wireframe
        
        self.composite_vao.render(moderngl.TRIANGLES)

        self.ctx.wireframe = False

            # not sure how moderngl handels buffer swapping
            # but removing this gives about 20fps more
            # and this also results in a flickering
            # switch buffers for rendering 
            #self.compute_buffer_a, self.compute_buffer_b = self.compute_buffer_b, self.compute_buffer_a
            #self.points_a, self.points_b = self.points_b, self.points_a

        self.render_ui()

    def render_depth_edges(self):

        offset_v = 1.0 / (self.wnd.size[1] * self.offset_factor)
        offset_h = 1.0 / (self.wnd.size[0] * self.offset_factor)

        # switch framebuffer
        self.current_fbo, self.back_fbo = self.back_fbo, self.current_fbo

        self.current_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        self.depth_edge_prog['depthTexture'].value = 0
        self.my_framebuffer.depth_attachment.use(location=0)
        self.depth_edge_prog['depth_contour_amp'] = self.depth_contour_amp
        self.depth_edge_prog['depth_contour_color'] = self.depth_contour_color
        self.depth_edge_prog['offset_h'] = offset_h
        self.depth_edge_prog['offset_v'] = offset_v
        self.depth_edge_prog['opaque'] = self.depth_opaque
        self.depth_edges_vao.render(moderngl.TRIANGLES)

        for i in range(self.depth_dilation_iterations):

            # switch framebuffer
            self.current_fbo, self.back_fbo = self.back_fbo, self.current_fbo

            self.current_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self.dilation_prog['edgeTexture'].value = 0
            self.back_fbo.color_attachments[0].use(location=0)
            self.dilation_prog['parameters'] = np.r_[1, offset_h, offset_v]
            self.dilation_vao.render(moderngl.TRIANGLES)

        if self.depth_blur:

            #self.ctx.blend_func = (moderngl.ONE_MINUS_SRC_ALPHA, moderngl.SRC_ALPHA)
            #self.ctx.blend_equation = moderngl.FUNC_ADD

            # switch framebuffer
            self.current_fbo, self.back_fbo = self.back_fbo, self.current_fbo

            self.current_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self.gaussian_prog['Texture'].value = 0
            self.back_fbo.color_attachments[0].use(location=0)
            self.gaussian_prog['offset_h'] = offset_h
            self.gaussian_prog['offset_v'] = offset_v
            self.gaussian_prog['dir'] = np.r_[1.0, 0.0]
            self.gaussian_prog['sigma'] = self.depth_sigma
            self.gaussian_prog['kernelSize'] = 5
            self.gaussia_vao.render(moderngl.TRIANGLES)

            # switch framebuffer
            self.current_fbo, self.back_fbo = self.back_fbo, self.current_fbo

            self.current_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self.gaussian_prog['Texture'].value = 0
            self.back_fbo.color_attachments[0].use(location=0)
            self.gaussian_prog['offset_h'] = offset_h
            self.gaussian_prog['offset_v'] = offset_v
            self.gaussian_prog['dir'] = np.r_[0.0, 1.0]
            self.gaussian_prog['sigma'] = self.depth_sigma
            self.gaussian_prog['kernelSize'] = 5
            self.gaussia_vao.render(moderngl.TRIANGLES)

        self.ctx.copy_framebuffer(self.depth_edges_fbo, self.current_fbo)

    def render_exen_edges(self):

        offset_v = 1.0 / (self.wnd.size[1] * self.offset_factor)
        offset_h = 1.0 / (self.wnd.size[0] * self.offset_factor)

        # switch framebuffer
        self.current_fbo, self.back_fbo = self.back_fbo, self.current_fbo

        ##### gaussian blur the exen texture before edge detection

        self.current_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.gaussian_prog['Texture'].value = 0
        self.my_framebuffer.color_attachments[1].use(location=0)
        self.gaussian_prog['offset_h'] = offset_h
        self.gaussian_prog['offset_v'] = offset_v
        self.gaussian_prog['dir'] = np.r_[1.0, 0.0]
        self.gaussian_prog['sigma'] = 20.0
        self.gaussian_prog['kernelSize'] = 5
        self.gaussia_vao.render(moderngl.TRIANGLES)

        # switch framebuffer
        self.current_fbo, self.back_fbo = self.back_fbo, self.current_fbo

        self.current_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        self.gaussian_prog['Texture'].value = 0
        self.back_fbo.color_attachments[0].use(location=0)
        self.gaussian_prog['offset_h'] = offset_h
        self.gaussian_prog['offset_v'] = offset_v
        self.gaussian_prog['dir'] = np.r_[0.0, 1.0]
        self.gaussian_prog['sigma'] = 20.0
        self.gaussian_prog['kernelSize'] = 5
        self.gaussia_vao.render(moderngl.TRIANGLES)

        #####

        # switch framebuffer
        self.current_fbo, self.back_fbo = self.back_fbo, self.current_fbo

        self.current_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)

        self.exen_edge_prog['explicitEncodingTexture'].value = 0
        self.back_fbo.color_attachments[0].use(location=0)
        self.exen_edge_prog['exen_contour_amp'] = self.exen_contour_amp
        self.exen_edge_prog['exen_contour_color'] = self.exen_contour_color
        self.exen_edge_prog['exen_number_contour_lines'] = self.exen_number_contour_lines
        self.exen_edge_prog['offset_h'] = offset_h
        self.exen_edge_prog['offset_v'] = offset_v
        self.exen_edge_prog['opaque'] = self.exen_opaque
        self.exen_edges_vao.render(moderngl.TRIANGLES)

        for i in range(self.exen_dilation_iterations):

            # switch framebuffer
            self.current_fbo, self.back_fbo = self.back_fbo, self.current_fbo

            self.current_fbo.use()
            self.ctx.clear(0.0, 0.0, 0.0, 0.0)
            self.dilation_prog['edgeTexture'].value = 0
            self.back_fbo.color_attachments[0].use(location=0)
            self.dilation_prog['parameters'] = np.r_[1, offset_h, offset_v]
            self.dilation_vao.render(moderngl.TRIANGLES)

        self.ctx.copy_framebuffer(self.exen_edges_fbo, self.current_fbo)
    

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
                            if self.loaded:
                                self.ens.swap(i, next)
                                self.swap()
                            imgui.reset_mouse_drag_delta()


                imgui.tree_pop()

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
            imgui.text(str(self.fps))
            _, self.play_speed = imgui.slider_float("Speed", self.play_speed, 0.00001, 0.5)
            _, self.bg_color = imgui.color_edit3("Background Color", *self.bg_color)
            _, self.transparency = imgui.slider_float("Transparency", self.transparency, 0.0, 1.0)
            _, self.point_size = imgui.slider_float("P", self.point_size, 1.0, 30.0)
            _, self.varying_size = imgui.checkbox("Varying Size", self.varying_size)
            _, self.wireframe = imgui.checkbox("Wireframe", self.wireframe)
            _, self.easyease = imgui.checkbox("Ease", self.easyease)

        optimal_transport, _ = imgui.collapsing_header("Optimal Transport", True)
        if optimal_transport:
            _, self.normalize_data = imgui.checkbox("Normalize Data", self.normalize_data)
            _, self.uniform_reference = imgui.checkbox("Uniform Reference", self.uniform_reference)
            _, self.reference_n = imgui.input_int("N", self.reference_n)
        
            _, self.ot_blur = imgui.input_float("Blur", self.ot_blur)
            _, self.ot_scaling = imgui.input_float("Scaling", self.ot_scaling)
            _, self.ot_trunctate = imgui.input_float("Truncate", self.ot_trunctate)
            _, self.ot_reach = imgui.input_float("Reach", self.ot_reach)

        comparison, _ = imgui.collapsing_header("Comparison", True)
        if comparison:
            _, self.color_distance = imgui.checkbox("Color Distance", self.color_distance)
            # contour 
            _, self.contour_overlay = imgui.checkbox("Ex Contour Overlay", self.contour_overlay)
            _, self.render_depth_contour = imgui.checkbox("Render Depth Contour", self.render_depth_contour)
            _, self.render_exen_contour = imgui.checkbox("Render Ex Contour", self.render_exen_contour)
            _, self.offset_factor = imgui.slider_float("Offset Factor", self.offset_factor, 0.1, 2.0)

            depth_contour_ui, _ = imgui.collapsing_header("Depth Contour", True)
            if depth_contour_ui:
                _, self.depth_contour_color = imgui.color_edit3("Depth Color", *self.depth_contour_color)
                _, self.depth_contour_amp = imgui.slider_float("CD", self.depth_contour_amp, 0.0, 100.0)
                _, self.depth_dilation_iterations = imgui.slider_int("D Dilation Iterations", self.depth_dilation_iterations, 0, 7)
                _, self.depth_blur = imgui.checkbox("Blur", self.depth_blur)
                _, self.depth_sigma = imgui.slider_float("Sigma", self.depth_sigma, 0.1, 10.0)
                _, self.depth_opaque = imgui.checkbox("D Opaque", self.depth_opaque)

            exen_contour_ui, _ = imgui.collapsing_header("Exen Contour", True)
            if exen_contour_ui:
                _, self.exen_contour_color = imgui.color_edit3("Explicit Encoding Color", *self.exen_contour_color)
                _, self.exen_contour_amp = imgui.slider_float("CEX", self.exen_contour_amp, 0.0, 5.0)
                _, self.exen_number_contour_lines = imgui.input_int("Ex Contour Lines", self.exen_number_contour_lines, 1.0, 100.0)
                _, self.exen_dilation_iterations = imgui.slider_int("E Dilation Iterations", self.exen_dilation_iterations, 0, 7)
                _, self.exen_opaque = imgui.checkbox("E Opaque", self.exen_opaque)

            _, self.filter_treshold = imgui.slider_float("Filter Treshold", self.filter_treshold, 0.0, 1.0)

        imgui.end()

        ##### slider

        wnd_size = self.wnd.size
        imgui.set_next_window_size(wnd_size[0], 115)
        imgui.set_next_window_position(0, wnd_size[1]-115)
        imgui.begin("State", False, flags=imgui.WINDOW_NO_COLLAPSE)
        _, self.lock_states = imgui.checkbox("Lock", self.lock_states)
        imgui.same_line()
        _, self.play = imgui.checkbox("Play", self.play)

        slider_width = wnd_size[0] - (wnd_size[0] * 0.02)

        if self.loaded:
            imgui.push_font(self.font)
            for i in range(1, self.number_of_files - 1):
                imgui.same_line(i * (slider_width / (self.number_of_files - 1)))
                imgui.text_colored("³", 1.0, 0.67, 0.4)
            imgui.pop_font()

        imgui.set_next_item_width(slider_width)
        _, self.transition_state = imgui.slider_float("Transition", self.transition_state, 0.0, 1.0)
        if self.loaded and (imgui.is_item_hovered() or imgui.is_item_active()):
            with imgui.begin_tooltip():
                text = round(self.transition_state / (1 / (self.number_of_files - 1)) - self.current_assignment, 3)
                imgui.text(f"State {text}.")

        imgui.set_next_item_width(slider_width)
        _, self.color_state = imgui.slider_float("Color", self.color_state, 0.0, 1.0)
        if self.loaded and (imgui.is_item_hovered() or imgui.is_item_active()):
            with imgui.begin_tooltip():
                text = round(self.color_state / (1 / (self.number_of_files - 1)) - self.current_assignment, 3)
                imgui.text(f"Color State {text}.")

        imgui.end()
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def run_debug(self):
        filelist = {"files": [
                                pathlib.Path(__file__).parents[3] / "data/loewe/lion3.ply",
                                pathlib.Path(__file__).parents[3] / "data/loewe/lion2.ply",
                                pathlib.Path(__file__).parents[3] / "data/loewe/lion1.ply"
                             ]         
                    }
        self.files = [str(file.resolve()) for file in filelist["files"]]
        self.normalize_data = True
        self.uniform_reference = False
        self.generic_run(filelist)

    def run_ot(self):
        filelist_path = util.create_tmp_dir() / "filelist.json"
        if filelist_path.is_file:
            with open(filelist_path, 'r') as infile:
                filelist = json.load(infile)
                self.generic_run(filelist)

    def generic_run(self, filelist):
        self.number_of_files = len(filelist['files']) + (1 if self.uniform_reference else 0)

        conf = {
            "octree_node_size": 1000,
            "normalize_data": self.normalize_data,
            "autograd": True,
            "sort_emd": False,
            "uniform_reference": self.uniform_reference,
            "reference_n": self.reference_n,
            #"accumulate_distance": self.accumulate_distance
        }

        self.ens = Ensemble(filelist, conf)
        self.ens.build()
        #self.ens.ot_sequential()
        conf = {
            "blur" : self.ot_blur,
            "scaling" : self.ot_scaling,
            "truncate" : self.ot_trunctate,
            "reach": self.ot_reach
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

    def swap(self):
        source_pos, target_pos = self.ens.compute_data

        self.compute_buffer_b = self.ctx.buffer(source_pos)
        self.source_buffer = self.ctx.buffer(source_pos)
        self.target_buffer = self.ctx.buffer(target_pos)

        self.points_b = self.ctx.vertex_array(
            self.prog, [self.compute_buffer_b.bind('in_position', 'in_color', layout='4f 4f')],
        )

    def get_transition_state(self):
        
        ts = self.transition_state * (self.number_of_files - 1) - self.current_assignment

        if self.easyease:
            return easeInOutCubic(ts)
        else:
            return ts
    
    def get_color_state(self):

        cs = self.color_state * (self.number_of_files - 1) - self.current_assignment

        if self.easyease:
            return easeInOutCubic(cs)
        else:
            return cs

def easeInOutCubic(x):
    return 4 * x * x * x if x < 0.5 else 1 - pow(-2 * x + 2, 3) / 2
    

if __name__ == '__main__':
    Renderer.run()