# GPGPU Inverse Distance Weighting
A parallel GPU version of the IDW interpolation algorithm [1] using matrix-vector multiplication.

* custom_version -> matrix vector multiplication using a thread for each scalar product

* cuBLAS_version -> cuBLAS matrix vector multiplication

Language: C, Framework: CUDA 8.0, cuBLAS v2

# Authors
Mario Ruggieri

e-mail: mario.ruggieri@studenti.uniparthenope.it

Livia Marcellino

e-mail: livia.marcellino@uniparthenope.it
  
# Installation and Usage 

`src/demo.cu` is an example of usage on random data. It launches the CPU and GPU algorithms and shows for both the execution time.

**Demo installation**
  ```
  cd src
  make
  ```
In this way you are making `demo` binary file
	
**Demo usage**

* 1th argument: type of usage (1 = file, 2 = random data)

* 2th argument: (type 1) known 3D points dataset file / (type 2) number of known values

* 3th argument : (type 1) 2D locations file / (type 2) number of values to interpolate

* 4th argument : number of locations per iteration

* 5th argument : number of CUDA block threads

Examples:

	./demo 1 dataset.txt locations1k.txt 1000 80
	./demo 2 1000 1000 500 80
	
* dataset.txt contains a dataset of 45147 3D points
* locations1k.txt contains 1000 2D locations to calculate values
* locations200k.txt contains 219076 2D locations to calculate values

# Version
This is a beta version made for academic purposes.
	
# Licensing
Please read LICENSE file.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

# References
[1] Shepard D. , A two-dimensional interpolation function for irregularly-spaced data, Proceedings of the 1968 ACM National Conference. pp. 517–524 
