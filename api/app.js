const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 7130;
const STATIC_DIR = path.join(__dirname, 'dist/v1/browser');

app.use(express.static(STATIC_DIR));

app.get('*', (req, res) => {
  res.sendFile(path.join(STATIC_DIR, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`3D scan running on port ${PORT}`);
});
