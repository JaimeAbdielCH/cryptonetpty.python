import os
import tornado.ioloop
import tornado.web
from handlers.secure_request import SecureRequestHandler

import firebase_admin
from firebase_admin import credentials

os.environ['USER_AGENT'] = 'cryptonetpty-backend'
os.environ["FIREBASE_AUTH_EMULATOR_HOST"] = "127.0.0.1:9099"  # Authentication emulator
# Initialize Firebase Admin SDK
cred = credentials.Certificate("d34gn-572ecca646.json")
firebase_admin.initialize_app(cred)


class HealthCheckHandler(SecureRequestHandler):
    def get(self):       
        self.write("API is running!")

def make_app():
    return tornado.web.Application([
        (r"/api/health", HealthCheckHandler), # Health check endpoint
    ], debug=True)

if __name__ == "__main__":
    app = make_app()
    app.listen(8000)
    print("Tornado server is running on http://localhost:8000")
    tornado.ioloop.IOLoop.current().start()