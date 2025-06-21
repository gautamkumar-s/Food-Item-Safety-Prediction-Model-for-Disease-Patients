from flask import Flask, render_template, request
from dia import predict_diabetes

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    output = "" 
    if request.method == 'POST':
        
        kcal = float(request.form['kcal'])
        protein = float(request.form['protein'])
        saturated_fat = float(request.form['saturated_fat'])
        fat = float(request.form['fat'])
        carbs = float(request.form['carbs'])
        sugar = float(request.form['sugar'])
        calcium = float(request.form['calcium'])

       
        input_data = [kcal, protein, fat, saturated_fat, carbs, sugar, calcium]
        result = predict_diabetes(input_data)

        if result == 1:
            output = "This is GOOD for diabetic patient."
        else:
            output = "This is NOT GOOD for diabetic patient."

    return render_template('dia.html', output=output)

if __name__ == '__main__':
    app.run(debug=True)
