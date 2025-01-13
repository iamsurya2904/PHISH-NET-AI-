import streamlit as st
import pyotp  # Used to generate random OTP
import time
import pyDes  # Used for DES encryption/decryption
import re  # Used for input validation

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~> Initialize session state <~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if "page" not in st.session_state:
    st.session_state.page = "login"  # Default to login page
if "current_user" not in st.session_state:
    st.session_state.current_user = None  # No user logged in by default
if "otp_secret" not in st.session_state:
    st.session_state.otp_secret = None  # To store OTP secret in session state


# ====================> Database Layer <=====================
class DatabaseLayer:
    def __init__(self):
        self.des_key = b"DESCRYPT"  # 8-byte key for DES (Not secure, used for demonstration)
        # Simulated encrypted user database
        self.users = {
            "Test_User@gmail.com": {
                "password": self.encrypt_password("Test1234"),
                "otp_secret": pyotp.random_base32()
            },
            "Abdullah_Example@gmail.com": {
                "password": self.encrypt_password("SWE314_Abood"),
                "otp_secret": pyotp.random_base32()
            },
            "Baseel_Example@gmail.com": {
                "password": self.encrypt_password("SWE314_Baseel"),
                "otp_secret": pyotp.random_base32()
            },
            "Meshal_Example@gmail.com": {
                "password": self.encrypt_password("SWE314_Meshal"),
                "otp_secret": pyotp.random_base32()
            },
             "Khaled_Example@gmail.com": {
                "password": self.encrypt_password("SWE314_Khaled"),
                "otp_secret": pyotp.random_base32()           
        },
             "Abojan_Example@gmail.com": {
                "password": self.encrypt_password("SWE314_Abojan"),
                "otp_secret": pyotp.random_base32()           
        },
        }

    def get_user(self, email):
        return self.users.get(email)

    def encrypt_password(self, password):
        # Encrypt the password using DES
        des = pyDes.des(self.des_key, pyDes.ECB, padmode=pyDes.PAD_PKCS5)
        encrypted_password = des.encrypt(password)
        return encrypted_password

    def verify_password(self, stored_password, provided_password):
        # Decrypt stored password and compare
        des = pyDes.des(self.des_key, pyDes.ECB, padmode=pyDes.PAD_PKCS5)
        decrypted_password = des.decrypt(stored_password).decode('utf-8') #decrypt the code after encryption
        return decrypted_password == provided_password


# ====================> Application Layer <=====================
class ApplicationLayer:
    def __init__(self, db_layer):
        self.db_layer = db_layer

    def validate_input(self, email, password):
        # Validate email and password format
        email_pattern = r"[^@]+@[^@]+\.[^@]+" 
        password_pattern = r"^[A-Za-z0-9@#$%^&+=_]{8,}$"  # Minimum 8 characters

        if not re.match(email_pattern, email):
            return False, "Invalid email format."
        if not re.match(password_pattern, password):
            return False, "Password must be at least 8 characters and include letters, numbers, and symbols."
        return True, ""

    def authenticate_user(self, email, password):
        # Authenticate the user
        user = self.db_layer.get_user(email)
        if user and self.db_layer.verify_password(user["password"], password):
            return True, user["otp_secret"]
        else:
            return False, None

    def generate_otp(self, otp_secret):
        # Generate OTP
        otp = pyotp.TOTP(otp_secret).now()
        return otp

    def verify_otp(self, otp_secret, otp):
        # Verify OTP
        return pyotp.TOTP(otp_secret).verify(otp, valid_window=1)

# ====================> Presentation Layer <=====================
def login_page(app_layer):
    st.markdown("<h1 style='text-align: center;'>MFA Authentication - Login</h1>", unsafe_allow_html=True)

    # Display test credentials
    st.markdown("<h3 style='color:grey;'>Test Login Credentials</h6>", unsafe_allow_html=True)
    st.markdown("<p style='color: grey;'>Email: Test_User@gmail.com</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:grey;'>Password: Test1234</p>", unsafe_allow_html=True)
    st.divider()

    # Input fields
    email = st.text_input("Email").strip()
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        # Input validation
        is_valid, message = app_layer.validate_input(email, password)
        if not is_valid:
            st.error(message)
            return

        # Authenticate user
        is_authenticated, otp_secret = app_layer.authenticate_user(email, password)
        if is_authenticated:
            otp = app_layer.generate_otp(otp_secret)  # Generate OTP
            st.session_state.current_user = email  # Save the logged-in user
            st.session_state.otp_secret = otp_secret  # Store OTP secret in session state

            st.success("OTP sent")
            st.toast(f"OTP: {otp}")  #send OTP via email/SMS (but our code only send it within the system for simplification perposes)
            time.sleep(4)
            st.session_state.page = "otp"  # Go to OTP page
            st.rerun()  # rerun to display OTP page
        else:
            st.error("Invalid credentials")


def otp_page(app_layer):
    st.markdown("<h1 style='text-align: center; color:grey;'>MFA Authentication - OTP Verification</h1>", unsafe_allow_html=True)
    st.divider()
    # Ensure the user is logged in
    if not st.session_state.current_user:
        st.warning("Please log in first!")
        st.session_state.page = "login"
        st.rerun()

    # OTP verification
    otp = st.text_input("Enter OTP")
    if st.button("Verify OTP"):
        if app_layer.verify_otp(st.session_state.otp_secret, otp):  # Verify OTP
            st.success("Login successful!")
            time.sleep(2.5)
            st.session_state.page = "main"  # Proceed to main page
            st.rerun()  # Rerun to display main page
        else:
            st.error("Invalid OTP")

    # Option to go back
    if st.button("Go Back"):
        st.session_state.page = "login"
        st.rerun()


def main_page():
    st.markdown("<h1 style='text-align: center; color:grey;'>Customer Banking System</h1>", unsafe_allow_html=True)
    st.divider()

    st.subheader("Account Summary")  # Simple Account Summary
    st.write("**Account Balance:** $12,345.67")
    st.write("**Savings Balance:** $8,210.45")

    # Simple Quick Actions
    st.subheader("Quick Actions")
    col1, col2, col3 = st.columns(3)  # split the features into 3 columns
    with col1:
        if st.button("Transfer Funds", use_container_width=True):
            st.toast("Transfer Fund feature coming soon!")
    with col2:
        if st.button("Pay Bills", use_container_width=True):
            st.toast("Bill payment feature coming soon!")
    with col3:
        if st.button("View Statements", use_container_width=True):
            st.toast("View Statement view feature coming soon!")

    st.divider()
    if st.button("Log Out", use_container_width=True):  # Logout Button
        st.session_state.page = "login"
        st.rerun()

# ====================> Navigation <=====================
# Initialize layers
db_layer = DatabaseLayer()
app_layer = ApplicationLayer(db_layer)

if st.session_state.page == "login":
    login_page(app_layer)
elif st.session_state.page == "otp":
    otp_page()
elif st.session_state.page == "main":
    main_page()