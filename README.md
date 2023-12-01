
### Dependencies: 
- download cuda
- have glfw on system (`yay glfw-x11`)
- have OpenGL

### Compile
- `nvcc main.cu glad.c -lglfw -o main`
### Run 
- `./main`

### How to use: 
- WASD to move 
- I and O to zoom In and Out

### facts:

- currently using naive implementation of cell update alg
- gets around 30 fps currently at startup when doing 2048x2048 cells


### PERFORMANCE HISTORY

- with raylib and a 2048x2048 canvas with no optimizations and NO CELL TRANSFORMING - getting 36 FPS, while 290 fps at -O3

- with glfw + OpenGL + copying CUDA data to SSBO with floats for cells, indexed in fragment shader and a 2048x2048 canvas with no optimizations and NO CELL TRANSFORMING - getting 120 FPS, while 350 fps at -O3

- as previous, but now CUDA data is sent directly to SSBO, shader does bit manipulation of 32 uints to turn them into bools - getting 800 FPS at NO CELL TRANSFORMING - while 730 fps with transforming, practically no difference for -O3.
