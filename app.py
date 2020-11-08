from flask import Flask, render_template

from src.business_logic.process_query import create_business_logic

app = Flask(__name__)

#import os
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/Admin/Desktop/stock_final_project/artful-athlete-292316-70398a8b80a6.json"

@app.route('/')
def index():
   return render_template('index.html')


@app.route('/', methods=['GET'])
def hello():
    return f'Hello you should use an other route:!\nEX: get_stock_val/<ticker>\n'


@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    bl = create_business_logic()
    prediction = bl.do_predictions_for(ticker)
    output = prediction
    #return prediction
    return render_template('results.html', output=output)

if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='localhost', port=8080, debug=True)


