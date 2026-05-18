import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import HardNoteApp from './HardNoteApp.jsx';
import './styles.css';

const mode = new URLSearchParams(window.location.search).get('mode');
const Root = mode === 'hard' ? HardNoteApp : App;

ReactDOM.createRoot(document.getElementById('root')).render(<Root />);
