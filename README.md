# Evolutionary Affinity Propagation

Evolutionary Affinity Propagation (EAP) is an evolutionary clustering method, seeking to cluster data collected at multiple time points while taking into account underlying dynamics and conserving temporal smoothness. 

The key distinctive features of EAP are:
1. EAP automatically determines the number of clusters at each time step.
2. By relying on certain consensus nodes introduced in the factor graph, EAP
accurately and effciently tracks clusters across time.
3. EAP identifies the global clustering solution by passing messages between the
nodes representing data at different time steps.
4. The EAP output provides the cluster membership for each data point at each
time instance.
5. The EAP framework allows having different numbers of clusters at different
time steps as well as data point insertions and deletions.


## Usage

`evolutionary_affinity_propagation.m` contains the Matlab code for EAP. <br>
`run_EAP.m` contains sample code for running EAP using the data in `sample_data.mat`. The 
description of the synthetic data can be found at the top of `run_EAP.m`.


## Reference

Arzeno, N.M. and Vikalo, H., 2017, March. Evolutionary affinity propagation. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 2681-2685). IEEE.


## Contact

Natalia Arzeno - narzeno@utexas.edu


## License

This project is licensed under the MIT License.

Copyright (c) 2019 Natalia Arzeno

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.