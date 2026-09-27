const React = require('react');
const ReactDOMServer = require('react-dom/server');
const sharp = require('sharp');
const {
  FaBrain, FaMicrochip, FaRulerCombined, FaVolumeUp, FaCalendarAlt,
  FaWallet, FaExclamationTriangle, FaBook, FaCamera, FaBullseye,
  FaProjectDiagram, FaChild, FaFan
} = require('react-icons/fa');

const icons = {
  brain: FaBrain, chip: FaMicrochip, ruler: FaRulerCombined, volume: FaVolumeUp,
  calendar: FaCalendarAlt, wallet: FaWallet, alert: FaExclamationTriangle,
  book: FaBook, camera: FaCamera, target: FaBullseye, diagram: FaProjectDiagram,
  person: FaChild, fan: FaFan,
};

async function run() {
  for (const [name, Icon] of Object.entries(icons)) {
    const svg = ReactDOMServer.renderToStaticMarkup(
      React.createElement(Icon, { size: 256, color: '#FFFFFF' })
    );
    await sharp(Buffer.from(svg)).png().toFile(`icons/${name}.png`);
    console.log('done', name);
  }
}
run();
