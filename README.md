# master-thesis
Master Thesis at TIØ4900 by Nick Askari and Simen Peder Stang.

Code is ran using a homemade distributed system created by us,
Find it at: https://github.com/nickaskari/compute-farm

## Setting up Virtual Environment (we are calling it "packages")

```sh
python -m venv packages 
```
Then activating (MAC OS)..
```sh
source packages/bin/activate
```
For Windows..
```sh
.\packages\Scripts\activate
```
## Handling packages
Add to requirements file after pip installing,
```sh
pip freeze > requirements.txt
```
Installing all libraries from requirements.txt,
```sh
pip install -r requirements.txt
```

TODO:
2. Testing the other models, and putting them in one systemized visiualization
  - Make less verbose, like a nice table or something
3. Combing the GANs with copulas
4. Add Multivariate GAN to the testing framework
5. Try to improve the individual GANs a little
6. Move on to Conditional GANs
