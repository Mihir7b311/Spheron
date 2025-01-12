// components/Navbar.js
import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Home, Calendar, Settings, Bell, HelpCircle, User } from 'lucide-react';
import '../styles/Navbar.css';

const Navbar = () => {
  const location = useLocation();

  return (
    <nav className="navbar">
      <div className="navbar-content">
        <div className="nav-left">
          <Link to="/">
            <button className={`nav-button ${location.pathname === '/' ? 'active' : ''}`}>
              <Home size={18} />
              Dashboard
            </button>
          </Link>
          <Link to="/schedules">
            <button className={`nav-button ${location.pathname === '/schedules' ? 'active' : ''}`}>
              <Calendar size={18} />
              My Schedules
            </button>
          </Link>
          <Link to="/settings">
            <button className={`nav-button ${location.pathname === '/settings' ? 'active' : ''}`}>
              <Settings size={18} />
              Settings
            </button>
          </Link>
        </div>
        
        <div className="nav-right">
          <button className="icon-button">
            <Bell size={20} />
            <span className="notification-badge">3</span>
          </button>
          <button className="icon-button">
            <HelpCircle size={20} />
          </button>
          <button className="profile-button">
            <User size={20} />
            <span>John Doe</span>
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;