
from OpenGL.GL import *

class Shader:
    def __init__(self, vp, fp):
        self.program = glCreateProgram()
        vs = self._c(GL_VERTEX_SHADER, open(vp).read())
        fs = self._c(GL_FRAGMENT_SHADER, open(fp).read())
        glAttachShader(self.program, vs); glAttachShader(self.program, fs); glLinkProgram(self.program)
        if glGetProgramiv(self.program, GL_LINK_STATUS) != GL_TRUE:
            raise RuntimeError(glGetProgramInfoLog(self.program).decode())
        glDeleteShader(vs); glDeleteShader(fs)
    def _c(self, t, src):
        sid = glCreateShader(t); glShaderSource(sid, src); glCompileShader(sid)
        if glGetShaderiv(sid, GL_COMPILE_STATUS) != GL_TRUE:
            raise RuntimeError(glGetShaderInfoLog(sid).decode())
        return sid
    def use(self): glUseProgram(self.program)
    def set_mat4(self, n, m): glUniformMatrix4fv(glGetUniformLocation(self.program, n.encode()), 1, GL_TRUE, m)
    def set_vec3(self, n, v): glUniform3f(glGetUniformLocation(self.program, n.encode()), float(v[0]), float(v[1]), float(v[2]))
    def set_float(self, n, v): glUniform1f(glGetUniformLocation(self.program, n.encode()), float(v))
    def set_int(self, n, v): glUniform1i(glGetUniformLocation(self.program, n.encode()), int(v))
    def set_vec3_array(self, name, arr):
        for i, v in enumerate(arr):
            loc = glGetUniformLocation(self.program, f"{name}[{i}]".encode())
            glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))
    def set_float_array(self, name, arr):
        for i, v in enumerate(arr):
            loc = glGetUniformLocation(self.program, f"{name}[{i}]".encode())
            glUniform1f(loc, float(v))
