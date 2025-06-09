def is_admin(api_key: str) -> bool:
    return api_key == "secret_admin_token"