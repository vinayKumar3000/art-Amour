from flask import Flask, render_template, url_for, request, session, redirect
from flask_pymongo import PyMongo
import bcrypt
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import imutils
import os

clusters = 5 # try changing it

app = Flask(__name__)
app.config['MONGO_URI']="mongodb+srv://18311A05P7:18311A05P7@cluster0.xobarbe.mongodb.net/db?retryWrites=true&w=majority"
mongodb_client = PyMongo(app)
db=mongodb_client.db

@app.route('/')
def index():
    if 'username' in session:
        return  render_template('front-end.html',username=session['username'])

    return render_template('login.html')

@app.route('/learn-blog')
def learn_blog():
     if 'username' in session:
        return  render_template('Learn-blog.html')

@app.route('/painting')
def painting():
    if 'username' in session:
        return  render_template('paintings.html')

@app.route('/mandalas')
def mandalas():
    if 'username' in session:
        return  render_template('mandalas.html')
    
@app.route('/upload',methods=['GET', 'POST'])
def upload():
     # If a form is submitted
    if request.method == "POST":
        
        # Unpickle classifier
        image2=request.files.get('img','')

        file=request.files['img']
        file.save(os.path.join('./', "received.jpg"))
    
        ## ML CODE
        img = cv2.imread('./received.jpg')
        org_img = img.copy()
        print('Org image shape --> ',img.shape)

        img = imutils.resize(img,height=200)
        print('After resizing shape --> ',img.shape)

        flat_img = np.reshape(img,(-1,3))
        print('After Flattening shape --> ',flat_img.shape)

        kmeans = KMeans(n_clusters=clusters,random_state=0)
        kmeans.fit(flat_img)

        dominant_colors = np.array(kmeans.cluster_centers_,dtype='uint')

        percentages = (np.unique(kmeans.labels_,return_counts=True)[1])/flat_img.shape[0]
        p_and_c = zip(percentages,dominant_colors)
        p_and_c = sorted(p_and_c,reverse=True)

        block = np.ones((50,50,3),dtype='uint')
   
        for i in range(clusters):
            plt.subplot(1,clusters,i+1)
            block[:] = p_and_c[i][1][::-1] # we have done this to convert bgr(opencv) to rgb(matplotlib) 
            plt.imshow(block)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel(str(round(p_and_c[i][0]*100,2))+'%')

        bar = np.ones((50,500,3),dtype='uint')
        plt.title('Proportions of colors in the image')
        start = 0
        i = 1
        for p,c in p_and_c:
            end = start+int(p*bar.shape[1])
            if i==clusters:
                bar[:,start:] = c[::-1]
            else:
                bar[:,start:end] = c[::-1]
            start = end
            i+=1

        rows = 1000
        cols = int((org_img.shape[0]/org_img.shape[1])*rows)
        img = cv2.resize(org_img,dsize=(rows,cols),interpolation=cv2.INTER_LINEAR)

        copy = img.copy()
        cv2.rectangle(copy,(rows//2-250,cols//2-90),(rows//2+250,cols//2+110),(255,255,255),-1)

        final = cv2.addWeighted(img,0.1,copy,0.9,0)
        cv2.putText(final,'Most Dominant Colors in the Image',(rows//2-230,cols//2-40),cv2.FONT_HERSHEY_DUPLEX,0.8,(0,0,0),1,cv2.LINE_AA)


        start = rows//2-220
        for i in range(5):
            end = start+70
            final[cols//2:cols//2+70,start:end] = p_and_c[i][1]
            cv2.putText(final,str(i+1),(start+25,cols//2+45),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),1,cv2.LINE_AA)
            start = end+20

      
        cv2.destroyAllWindows()
        cv2.imwrite(os.path.join('./static/img/', image2.filename),final)
        output="./static/img/"+image2.filename
    else:
        output="./static/img/default.jpg"
        
    return render_template("upload.html",output=output)


@app.route('/login', methods=['POST'])
def login():
 
    login_user = db.users.find_one({'name' : request.form['username']})

    if login_user:
        if bcrypt.checkpw(request.form['password'].encode('utf-8'), login_user['password']):
            session['username'] = request.form['username']
            return redirect(url_for('index'))

    return 'Invalid username/password combination'

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        existing_user = db.users.find_one({'name' : request.form['username']})

        if existing_user is None:
            hashpass = bcrypt.hashpw(request.form['password'].encode('utf-8'), bcrypt.gensalt())
            db.users.insert_one({'name' : request.form['username'], 'password' : hashpass})
            session['username'] = request.form['username']
            return redirect(url_for('index'))
        
        return 'That username already exists!'

    return render_template('signup.html')

if __name__ == '__main__':
    app.secret_key = 'mysecret'
    app.run(debug=True)