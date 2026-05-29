const { Pool } = require("pg");

const pool = new Pool({
  connectionString:
    "postgresql://postgres:[konvo@123lamiya]@db.jaxdnylknkvuiwvoehhx.supabase.co:5432/postgres",
  ssl: {
    rejectUnauthorized: false,
  },
});

module.exports = pool;