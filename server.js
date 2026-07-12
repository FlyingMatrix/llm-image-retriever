const express = require('express');
const { exec } = require('child_process');
const path = require('path');
const app = express();
const PORT = 4000;

app.use(express.json());
// Serve static frontend files (HTML/CSS) from a "public" folder
app.use(express.static('public'));

// 1. Search endpoint
app.get('/api/search', (req, res) => {
    const userQuery = req.query.q;
    if (!userQuery) {
        return res.status(400).json({ error: "Missing query parameter 'q'" });
    }

    // Safely wrap the query in quotes to pass to the CLI python script
    const command = `python image_retriever_ui.py search "${userQuery.replace(/"/g, '\\"')}"`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            return res.status(500).json({ error: error.message });
        }
        
        try {
            // Parse the JSON string printed by your Python script
            const result = JSON.parse(stdout.trim());
            res.json(result);
        } catch (e) {
            res.status(500).json({ error: "Failed to parse Python output", raw: stdout });
        }
    });
});

// 2. Image serving endpoint 
// Because your images live outside the web directory, Node handles sending them securely
app.get('/api/image', (req, res) => {
    const imagePath = req.query.path;
    if (!imagePath) return res.status(400).send("Path required");
    
    // Sends the local image file directly to the browser view
    res.sendFile(path.resolve(imagePath));
});

app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});