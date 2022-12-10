from flask import Flask, jsonify, request, render_template, make_response, session, redirect, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user
from flask_cors import CORS
import os
from view import mainlogo, main1
from control.playlist_management import Playlist

os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

app = Flask(__name__)
CORS(app)
app.secret_key = '1'

app.register_blueprint(mainlogo.mainlogo, url_prefix='/mainlogo')
app.register_blueprint(main1.main1, url_prefix='/main1')

@app.route('/')
def main():
    return redirect(url_for('mainlogo.main'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000')