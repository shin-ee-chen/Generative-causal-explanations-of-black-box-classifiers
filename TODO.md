# TO DO: 
- [x] Set up clear and definitive project structure
- [x] Clean up code, and write some documentation
- [ ] Introduce structure for including other encoders/decoders
    * Preferably these are all handled using a properly structured arg-parser
- [ ] Check training bottlenecks
    - Latent variable sweep takes too long (2x training) and time increases over epoches
    - For actual training, the approximate computation of the mutual information is likely the bottle-neck
        - Better approximation methods exist
        - More efficient versions of this algorithm can likely be found (e.g. chunked matrix calculations, Cython)
- [ ] Alter sweep code to allow for reproducing other figures
- [ ] Write code in ./experiments/ for producing latent variable sweeps from pre-trained models and ./utils/