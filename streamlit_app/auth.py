# streamlit_app/auth.py
def authenticate_user(username, password):
    if username == "admin" and password == "admin":
        return "admin"
    else:
        return "guest"