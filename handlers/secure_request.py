import tornado
from firebase_admin import auth


class SecureRequestHandler(tornado.web.RequestHandler):
    def prepare(self):
        # Extract the Firebase token from the Authorization header
        auth_header = self.request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            id_token = auth_header.split("Bearer ")[1]
        else:
            self.set_status(401)
            self.finish({"error": "Missing or invalid Authorization header"})
            return
        
        # Verify the Firebase JWT token
        try:
            decoded_token = auth.verify_id_token(id_token)
            self.user = decoded_token  # Store the user info for later use
        except Exception as e:
            self.set_status(401)
            self.finish({"error": "Invalid token", "details": str(e)})
            return