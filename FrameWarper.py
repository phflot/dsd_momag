import numpy as np
import moderngl


class FrameWarper:
    """
    This object handles the forward warping via a GPU shader program. The main offered function is warp_image
    which forward-warps the inputted image.
    """
    def __init__(self, image_size):
        """
        :param image_size: tuple of int, size of the images which are to be warped

        this function initializes the shader program and all necessary vertex and fragment buffers
        """
        self.image_size = image_size
        self.strip_indices = self.generate_triangle_strip_index_array()
        self.ctx = moderngl.create_context(standalone=True, require=330)
        self.ctx.enable(moderngl.DEPTH_TEST)

        self.prog = self.ctx.program(
            vertex_shader="""
                #version 330

                in vec2 in_vert;
                in vec3 in_color;
                in float in_depth;
                out vec3 v_color;


                void main() {
                    v_color = in_color;
                    gl_Position = vec4(in_vert, in_depth, 1.0);
                }
            """,
            fragment_shader="""
                #version 330

                in vec3 v_color;

                out vec3 f_color;

                void main() {
                    f_color = v_color;
                }
            """,
        )
        dummy_vertices = self.get_dummy_vertices()
        self.vertex_buffer = self.ctx.buffer(dummy_vertices)
        self.vertex_array = self.ctx.vertex_array(self.prog, self.vertex_buffer, 'in_vert', 'in_color', 'in_depth')
        self.frame_buffer = self.ctx.simple_framebuffer(image_size[::-1])

    def get_dummy_vertices(self):
        """
        :return: bytearray, creates a dummy array in the correct form to feed to the vertex buffer.
        """
        image = np.zeros((*self.image_size, 3))
        xx, yy = np.meshgrid(np.linspace(-1, 1, self.image_size[1], endpoint=True), np.linspace(-1, 1, self.image_size[0], endpoint=True))
        displacements = np.stack((xx, yy), axis=2)
        depth = np.zeros(self.image_size)
        return self.get_into_vertex_buffer_shape(image, displacements, depth)

    def get_into_vertex_buffer_shape(self, image, displacements, depth):
        """
        :param image: np.ndarray(n, m, 3) of float, normalized rgb image which is to be wared
        :param displacements: np.ndarray(n, m, 2) of float, target positions for each pixel
        :param depth:  np.ndarray(n, m) of float, depth value indicating which parts of the displaced image should be
                                                  drawn on top of others
        :retuen bytearray, the vertex buffer information carrying the input parameters
        """
        vertices = np.concatenate((displacements, image, depth[:, :, None]), axis=2)
        return vertices[self.strip_indices[:, 0], self.strip_indices[:, 1]].astype('f4').tobytes()

    def generate_triangle_strip_index_array(self):
        """
        :return: np.ndarray(k, 2) of float, array containing the vertex coordinates of the triangle strip vertex
                                            fed to the shader prgram. This defines the triangluation which is used
        """
        out_indices_x = []
        out_indices_y = []
        for i in range(self.image_size[1]-1):
            is_reversed = int((i % 2) * (-2) + 1)
            out_indices_x.append(np.arange(self.image_size[0]).repeat(2)[::is_reversed])
            out_indices_y.append(np.tile(np.array([i, i+1]), self.image_size[0]))

        out_indices_x = np.concatenate(out_indices_x).astype(np.int)
        out_indices_y = np.concatenate(out_indices_y).astype(np.int)
        return np.stack((out_indices_x, out_indices_y), axis=1)

    def read_frame_buffer(self):
        """
        :return: np.ndarray(n, m, 3) of uint8, returns the current rgb image stored in the frame buffer
        """
        return np.frombuffer(self.frame_buffer.read(), 'uint8').reshape(self.image_size[0], self.image_size[1], -1)

    def pixel_to_screenspace_coords(self, displacements):
        """
        :param displacements: np.ndarray(n, m, 2) of float, pixel displacement given in pixel coordinates
        :return: np.ndarray(n, m, 2) of float, pixel displacement given in texture coordinates (from -1 to 1 in both ax)
        """
        out = np.zeros_like(displacements)
        out[:, :, 1] = displacements[:, :, 0] * (2.0 / (self.image_size[0] - 1)) - 1.0
        out[:, :, 0] = displacements[:, :, 1] * (2.0 / (self.image_size[1] - 1)) - 1.0
        return out

    def warp_image(self, image, displacements, depth):
        """
        :param image: np.ndarray(n, m, 3) of float, normalized float image which should be warped
        :param displacements: np.ndarray(n, m, 2) of float, pixel destinations in pixel coordinates
        :param depth: np.ndarray(n, m) of float, depth values for each region indicating what parts of the image
                                                 should be drawn on top of others
        :return: np.ndarray(n, m, 3) of uint8, rgb 8 bit warped image
        """
        assert image.dtype == displacements.dtype == depth.dtype == np.float
        assert np.all(np.logical_and(image <= 1.0, image >= 0.0))
        assert image.shape[:2] == displacements.shape[:2] == depth.shape == self.image_size

        displacements = self.pixel_to_screenspace_coords(displacements)
        if depth.max() != depth.min():
            depth = -0.99 * (depth - depth.min()) / (depth.max() - depth.min())

        self.frame_buffer.use()
        self.frame_buffer.clear(0.0, 0.0, 0.0, 0.0)

        self.vertex_buffer.write(self.get_into_vertex_buffer_shape(image, displacements, depth))

        self.vertex_array.render(moderngl.TRIANGLE_STRIP)

        return self.read_frame_buffer()[:, :, :3]
