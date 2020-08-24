# INSTALL file for 20 News groups classifier

## Install

```shell
git clone git@github.com:gandalf012/SentenceBert-20newsgroup-Classifier.git
cd SentenceBert-20newsgroup-Classifier
```
### Download Fune tuned **distilbert-base-nli-mean-tokens** 20 news group dataset
From [Google Drive](https://drive.google.com/file/d/1VjYGZasx9sEuJ2u9DCirb8L2wdIYIcsM/view?usp=sharing)
download **fine-TripletLoss-20newsdistilbert-base-nli-mean-tokens-2020-08-18_20-39-24** directory and unzip in models directory

* Create venv
```shell
python3 -m venv venv
```
* Activate virtual env
```shell
python3 -m venv venv
```
* Install requirements
```shell
pip install -r requirements.txt
```
### Docker
```
docker build -t "20-news-classifier" .
./rundocker
```
### Web app
```
streamlit run interface.py
```


## Run development server

* Activate venv
```shell
source venv/bin/activate
```

* Run flask server
```shell
export FLASK_APP=app.py
flask run -h 0.0.0.0
```
## Examples for demo

1. It was a 2-door sports car, looked to be from the late 60s early 70s. It was called a Bricklin. The doors were really small. 
2. my mac plus finally gave up the ghost this weekend after starting life as a 512k way back in 1985.
3. Do you have Weitek's address/phone number?  I'd like to get some information about this chip
4. My understanding is that the 'expected errors' are basically known bugs in the warning system software - things are checked that don't have the right values in yet because they aren't set till after launch, and suchlike.
5. There were a few people who responded to my request for info on treatment for astrocytomas through email, whom I couldn't thank directly because of mail-bouncing probs 