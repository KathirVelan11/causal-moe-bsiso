// Minimal PNG size reader (IHDR width/height).
const fs = require("fs");
exports.imageSize = (file) => {
  const b = fs.readFileSync(file);
  return { width: b.readUInt32BE(16), height: b.readUInt32BE(20) };
};
