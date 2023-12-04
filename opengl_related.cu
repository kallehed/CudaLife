#include "include/glad/glad.h"
#include <cstdio>
#include <cstdlib>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "project_header.cuh"

unsigned int get_program() {
  const char *const vertex_shader_source =
      "#version 460\n"
      "float x_pos[6] = float[6](-1.f, -1.f, 1.f, -1.f, 1.f, 1.f); \n"
      "float y_pos[6] = float[6](-1.f, 1.f, 1.f, -1.f, 1.f, -1.f); \n"
      "void main()\n"
      "{\n"
      " float x = x_pos[gl_VertexID];"
      " float y = y_pos[gl_VertexID];"
      " gl_Position = vec4(x, y, 0.f, 1.0);\n"
      "}\0";
  unsigned int vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_shader_source, NULL);
  glCompileShader(vertex_shader);
  {
    int success;
    char infoLog[512];
    glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(vertex_shader, 512, NULL, infoLog);
      printf("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n %s\n", infoLog);
    }
  }
  // clang-format off
  const char *const fragment_shader_source =
      "#version 460 core\n"
      "out vec4 FragColor;\n"
      "uniform vec4 u_pos_and_scale;"
      "layout (std430, binding = 0) buffer Colors {\n"
      "  uint color[];\n" 
      "};\n"
      "void main() {\n"
      "  int x = int((gl_FragCoord.x + u_pos_and_scale.x)/u_pos_and_scale.z);\n"
      "  int y = int((gl_FragCoord.y + u_pos_and_scale.y)/u_pos_and_scale.w);\n"
      "  uint idx = x + y * " STR(WIDTH) ";\n"
      "  uint block = idx / 4;\n"
      "  uint byte = idx % 4;\n"
      "  uint col4 = color[block];\n"
      "  uint mask = (0x000000FF << (byte * 8));\n"
      "  uint colbool =  mask & col4;\n"
      "  float col = float(colbool);\n"
      "  if (x >= " STR(WIDTH) " || y >= " STR(HEIGHT) " || x < 0 || y < 0 ) {col = 0.5;}"
      "  FragColor = vec4(vec3(float(col)), 1.0f);\n"
      "}\n";
  // clang-format on
  unsigned int fragment_shader;
  fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_source, NULL);
  glCompileShader(fragment_shader);
  {
    int success;
    char infoLog[512];
    glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &success);
    if (!success) {
      glGetShaderInfoLog(fragment_shader, 512, NULL, infoLog);
      printf("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n %s\n", infoLog);
    }
  }
  unsigned int shader_program;
  shader_program = glCreateProgram();
  glAttachShader(shader_program, vertex_shader);
  glAttachShader(shader_program, fragment_shader);
  glLinkProgram(shader_program);
  {
    int success;
    char infoLog[512];
    glGetProgramiv(shader_program, GL_LINK_STATUS, &success);
    if (!success) {
      glGetProgramInfoLog(shader_program, 512, NULL, infoLog);
      printf("ERROR::PROGRAM::COMPILATION_FAILED\n %s\n", infoLog);
    }
  }
  glDeleteShader(vertex_shader);
  glDeleteShader(fragment_shader);
  return shader_program;
}

// slow, copies using OpenGL, inits on CPU
void randomize_world(unsigned int SSBO) {
  unsigned char *data = (unsigned char *)malloc(WORLD_BYTES);
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      unsigned char value;
      if (i == 0 || i == HEIGHT - 1 || j == 0 || j == WIDTH - 1) {
        value = 0;
      } else {
        value = rand() % 2;
      }
      data[i * WIDTH + j] = value;
    }
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
  glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, WORLD_BYTES, data);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  free(data);
}

void terminate_all_life_in_world(unsigned int SSBO) {
  unsigned char *data = (unsigned char *)malloc(WORLD_BYTES);
  for (int i = 0; i < HEIGHT; ++i) {
    for (int j = 0; j < WIDTH; ++j) {
      data[i * WIDTH + j] = 0;
    }
  }
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
  glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, WORLD_BYTES, data);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  free(data);
}

void world_set_cell(unsigned int SSBO, int x, int y,
                           unsigned char data) {
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, SSBO);
  glBufferSubData(GL_SHADER_STORAGE_BUFFER, x + y * WIDTH, 1, &data);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}
