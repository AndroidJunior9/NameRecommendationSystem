from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

file = open("model.pkl","rb")
model = pickle.load(file)
file.close()


    
@app.route("/",methods = ["GET","POST"])
def hello_world():
    popularity_map = {'very low': 0, 'low': 1, 'medium': 2, 'high': 3, 'very high': 4}
    if request.method == "POST":
        data = request.form.to_dict()
        gender = 0 if data["gender"]=="girl" else 1
        trending = True if data["trending"]=="yes" else False
        nameLength = data["nameLength"]
        firstLetter = ord(data["firstLetter"].lower())-97
        genderNeutral = True if data["genderNeutral"]=="yes" else False
        popularity = popularity_map[data["popularity"]]
        name = model.predict([[gender,trending,nameLength,firstLetter,genderNeutral,popularity]])
        return jsonify({'name': name[0]})
    
    #1.Gender
    #2.Trending
    #3.NameLength
    #4.First Letter
    #5.Gender Neutral
    #6.Popularity
    return render_template("index.html")
    #return "The name is "+str(model.predict([inputfeatures])[0])
if __name__ == "__main__":
    app.run(debug=True)