from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Für admin
print(pwd_context.hash("admin"))
# Für user
print(pwd_context.hash("user"))
# Für guest
print(pwd_context.hash("guest"))