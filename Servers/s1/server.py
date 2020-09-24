from flask import Flask, jsonify

app = Flask(__name__)

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol', 
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web', 
        'done': False
    }
]

@app.route('/api/v1/objective_function', methods=['GET'])
def get_objective_function():
    return jsonify({'objective_function': 1})

@app.rout('/api/v1/algorithm', methods=['GET'])
def get_algorithm():
    return jsonify({'alg': 1})

if __name__ == '__main__':
    app.run(debug=True)
