const express = require('express');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const mysql = require('mysql2/promise');
const router = express.Router();

const SECRET_KEY = 'your_secret_key';
const db = mysql.createPool({
    host: '127.0.0.1',
    user: 'root',
    password: 'Prateush@04',
    database: 'gpudb',
    port: 3306,
});

// Add authentication middleware
const authenticateToken = (req, res, next) => {
    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) return res.status(401).json({ message: 'No token provided' });

    jwt.verify(token, SECRET_KEY, (err, user) => {
        if (err) return res.status(403).json({ message: 'Invalid token' });
        req.user = user;
        next();
    });
};

// Modified register endpoint
router.post('/register', async (req, res) => {
    const { name, email, password } = req.body;

    try {
        const existingUser = await db.query('SELECT * FROM users WHERE email = ?', [email]);
        if (existingUser[0].length) {
            return res.status(400).json({ message: 'Email already registered' });
        }

        const passwordHash = await bcrypt.hash(password, 10);
        await db.query('INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)', 
            [name, email, passwordHash]
        );

        res.status(201).json({ message: 'Registration successful' });
    } catch (err) {
        console.error('Registration error:', err);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Modified login endpoint
router.post('/login', async (req, res) => {
    const { email, password } = req.body;
    console.log('Login api called');
    try {
        const [users] = await db.query('SELECT * FROM users WHERE email = ?', [email]);
        const user = users[0];

        if (!user || !await bcrypt.compare(password, user.password_hash)) {
            return res.status(401).json({ message: 'Invalid credentials' });
        }

        const token = jwt.sign(
            { id: user.id, name: user.name, email: user.email },
            SECRET_KEY,
            { expiresIn: '24h' }
        );

        res.json({
            token,
            user: {
                id: user.id,
                name: user.name,
                email: user.email
            }
        });
    } catch (err) {
        console.error('Login error:', err);
        res.status(500).json({ message: 'Internal server error' });
    }
});

// Verify token endpoint
router.get('/verify', authenticateToken, (req, res) => {
    console.log('Verify called');
    res.json({ user: req.user });
});

// Update user endpoint
router.put('/users/:id', authenticateToken, async (req, res) => {
    if (req.user.id !== parseInt(req.params.id)) {
        return res.status(403).json({ message: 'Unauthorized' });
    }

    const { name, email } = req.body;
    try {
        await db.query('UPDATE users SET name = ?, email = ? WHERE id = ?', 
            [name, email, req.params.id]
        );
        res.json({ message: 'Profile updated successfully' });
    } catch (err) {
        console.error('Update error:', err);
        res.status(500).json({ message: 'Internal server error' });
    }
});

module.exports = router;