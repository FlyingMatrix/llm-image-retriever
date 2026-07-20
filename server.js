const express = require('express');
const http = require('http'); // Required for Socket.IO integration
const { Server } = require('socket.io'); // Socket.IO
const { execFile, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');
const session = require('express-session');

const app = express();
const server = http.createServer(app); // Create HTTP server wrapped around Express
const io = new Server(server); // Attach Socket.IO to the server
const PORT = 4000;

// Define session middleware so it can be shared with Socket.IO
const sessionMiddleware = session({
    secret: 'image-retriever-secret-key', // Replace with a random secret in production
    resave: false,
    saveUninitialized: true,
    cookie: { 
        maxAge: 24 * 60 * 60 * 1000 // Session valid for 24 hours
    }
});

// Session Middleware
app.use(sessionMiddleware);

// Share session middleware with Socket.IO requests
io.engine.use(sessionMiddleware);

app.use(express.json());
app.use(express.static('public'));

// Real-Time Socket Connections & Disconnections Log
io.on('connection', (socket) => {
    const req = socket.request;
    const userId = req.session ? req.session.id : socket.id;

    console.log(`🟢 [User ${userId}] Connected (Socket: ${socket.id})`);

    socket.on('disconnect', () => {
        console.log(`🔴 [User ${userId}] Disconnected (Tab/Window Closed)`);
    });
});

// Endpoint to fetch the current user's unique ID on the frontend
app.get('/api/user-info', (req, res) => {
    res.json({ userId: req.sessionID });
});

// Logout Endpoint
app.post('/api/logout', (req, res) => {
    const userId = req.sessionID;

    req.session.destroy((err) => {
        if (err) {
            console.error(`[User ${userId}] Logout Error:`, err);
            return res.status(500).json({ error: "Failed to log out" });
        }

        console.log(`[User ${userId}] Logged out`);
        res.clearCookie('connect.sid'); // Clear Express session cookie
        res.json({ status: 'success', message: `User ${userId} logged out successfully` });
    });
});

// Search Endpoint (Multi-user safe)
app.get('/api/search', (req, res) => {
    const userQuery = req.query.q;
    const userId = req.sessionID;

    if (!userQuery) {
        return res.status(400).json({ error: "Missing query parameter 'q'" });
    }

    console.log(`[User ${userId}] Initiating search: "${userQuery}"`);

    const pythonArgs = ['image_retriever_ui.py', 'search', userQuery];

    execFile('python', pythonArgs, (error, stdout, stderr) => {
        if (error) {
            console.error(`[User ${userId}] Search Error:`, stderr);
            return res.status(500).json({ error: error.message });
        }

        try {
            const result = JSON.parse(stdout.trim());
            res.json({ userId, result });
        } catch (e) {
            console.error(`[User ${userId}] Parsing Error:`, stdout);
            res.status(500).json({
                error: "Failed to parse Python output",
                raw: stdout
            });
        }
    });
});

// Image Serving Endpoint
app.get('/api/image', (req, res) => {
    const imagePath = req.query.path;
    if (!imagePath) {
        return res.status(400).send("Path required");
    }

    res.sendFile(path.resolve(imagePath));
});

// Open Folder Endpoint
app.post('/api/open-folder', (req, res) => {
    const targetPath = req.query.path;
    if (!targetPath) {
        return res.status(400).json({
            status: 'error',
            message: 'No path provided'
        });
    }

    const resolved = path.resolve(targetPath);

    if (!fs.existsSync(resolved)) {
        return res.status(404).json({
            status: 'error',
            message: 'File does not exist.'
        });
    }

    let child;

    try {
        if (process.platform === 'win32') {
            child = spawn('explorer.exe', [`/select,${resolved}`], {
                detached: true,
                windowsHide: false,
                stdio: 'ignore'
            });
        } else if (process.platform === 'darwin') {
            child = spawn('open', ['-R', resolved], {
                detached: true,
                stdio: 'ignore'
            });
        } else {
            child = spawn('xdg-open', [path.dirname(resolved)], {
                detached: true,
                stdio: 'ignore'
            });
        }

        child.on('error', (err) => {
            console.error('Failed to launch file explorer:', err);
        });

        child.unref();

        res.json({ status: 'success' });
    } catch (err) {
        console.error(err);
        res.status(500).json({
            status: 'error',
            message: err.message
        });
    }
});

// Listen using server.listen instead of app.listen
server.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});
