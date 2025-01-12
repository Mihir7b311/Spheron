// App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import DashboardPage from './pages/DashboardPage';
import SchedulePage from './pages/SchedulePage';
import './styles/App.css';

const App = () => {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/schedules" element={<SchedulePage />} />
          <Route path="/settings" element={
            <div className="p-8">
              <h1 className="text-2xl text-white">Settings Page</h1>
            </div>
          } />
        </Routes>
      </div>
    </Router>
  );
};

export default App;