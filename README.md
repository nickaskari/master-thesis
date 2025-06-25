# master-thesis
Master Thesis at TIØ4900 by Nick Askari and Simen Peder Stang.

Code was at times ran using a homemade distributed system created by us,
Find it at: https://github.com/nickaskari/compute-farm

After which we needed more compute, much of the development environment was migrated to Google Collab.
This repository contains the key notebooks, models and files which we used. Hence to run the Google Collab notebooks, and environment using google disk must be done.
For help, or curisoity in doing so yourself, contact us at: nickask12@gmail.com.

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
