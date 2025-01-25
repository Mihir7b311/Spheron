import React from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Home, Calendar, Settings, Bell, HelpCircle, User, LogOut } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import '../styles/Navbar.css';

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="navbar">
      <div className="navbar-content">
        <div className="nav-left">
          <Link to="/dashboard">
            <button className={`nav-button ${location.pathname === '/dashboard' ? 'active' : ''}`}>
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
          <div className="flex items-center gap-2">
            <button className="profile-button">
              <User size={20} />
              <span>{user?.name || 'User'}</span>
            </button>
            <button 
              onClick={handleLogout}
              className="icon-button text-red-400 hover:bg-red-500/10"
            >
              <LogOut size={20} />
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;