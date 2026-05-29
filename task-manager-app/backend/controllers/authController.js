const pool = require("../db/db");
const jwt = require("jsonwebtoken");

exports.signup = async (req, res) => {
  try {
    const { name, phone } = req.body;

    const otp = "123456";

    const user = await pool.query(
      "INSERT INTO users(name, phone, otp) VALUES($1,$2,$3) RETURNING *",
      [name, phone, otp]
    );

    res.json({
      message: "OTP Sent",
      otp,
      user: user.rows[0],
    });
  } catch (error) {
    res.status(500).json({
      error: error.message,
    });
  }
};

exports.verifyOtp = async (req, res) => {
  try {
    const { phone, otp } = req.body;

    const result = await pool.query(
      "SELECT * FROM users WHERE phone=$1",
      [phone]
    );

    const user = result.rows[0];

    if (!user || user.otp !== otp) {
      return res.status(400).json({
        message: "Invalid OTP",
      });
    }

    const token = jwt.sign(
      { id: user.id },
      "SECRET_KEY"
    );

    res.json({
      token,
      user,
    });
  } catch (error) {
    res.status(500).json({
      error: error.message,
    });
  }
};