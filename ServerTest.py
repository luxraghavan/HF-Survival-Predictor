from flask import Flask

app = Flask(__name__)

@app.route('/')

@app.route('/index')
def home():
    return 'Hi, Welcome to Flask!!'

if __name__ == '__main__':
    print('Starting server ...')

    app.run()
