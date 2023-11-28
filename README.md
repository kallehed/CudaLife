
### Dependencies: 
- download cuda
- have glfw on system (`yay glfw-x11`)
- have OpenGL

### Compile
- `nvcc main.cu -lglfw -o main`
### Run 
- `./main`


### facts:

- currently using naive implementation of cell update alg
- gets around 30 fps currently at startup when doing 2048x2048 cells


### FACTS TO USE 

- with raylib and a 2048x2048 canvas with no optimizations and NO CELL TRANSFORMING - getting 36 FPS, while 290 fps at -O3
