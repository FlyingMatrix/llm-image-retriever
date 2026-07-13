const express = require('express');
const { exec, spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = 4000;

app.use(express.json());
app.use(express.static('public'));

// 1. Search endpoint
app.get('/api/search', (req, res) => {
    const userQuery = req.query.q;
    if (!userQuery) {
        return res.status(400).json({ error: "Missing query parameter 'q'" });
    }

    const command = `python image_retriever_ui.py search "${userQuery.replace(/"/g, '\\"')}"`;

    exec(command, (error, stdout, stderr) => {
        if (error) {
            console.error(stderr);
            return res.status(500).json({ error: error.message });
        }

        try {
            const result = JSON.parse(stdout.trim());
            res.json(result);
        } catch (e) {
            console.error(stdout);
            res.status(500).json({
                error: "Failed to parse Python output",
                raw: stdout
            });
        }
    });
});

// 2. Image serving endpoint
app.get('/api/image', (req, res) => {
    const imagePath = req.query.path;
    if (!imagePath) {
        return res.status(400).send("Path required");
    }

    res.sendFile(path.resolve(imagePath));
});

// 3. Open Folder endpoint
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
            // Reveal the selected file in Windows Explorer
            child = spawn('explorer.exe', [`/select,${resolved}`], {
                detached: true,
                windowsHide: false,
                stdio: 'ignore'
            });
        } else if (process.platform === 'darwin') {
            // Reveal the file in Finder
            child = spawn('open', ['-R', resolved], {
                detached: true,
                stdio: 'ignore'
            });
        } else {
            // Open the containing folder on Linux
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

app.listen(PORT, () => {
    console.log(`Server is running at http://localhost:${PORT}`);
});