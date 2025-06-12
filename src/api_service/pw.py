from passlib.context import CryptContext
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# for admin
print(pwd_context.hash("admin"))
# for user
print(pwd_context.hash("user"))
# for guest
print(pwd_context.hash("guest"))