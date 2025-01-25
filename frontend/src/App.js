import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { ProtectedRoute, PublicRoute } from './components/ProtectedRoute';
import Navbar from './components/Navbar';
import LoginPage from './components/auth/LoginPage';
import RegisterPage from './components/auth/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import SchedulePage from './pages/SchedulePage';
import './styles/App.css';

const App = () => {
  return (
    <Router>
      <AuthProvider>
        <div className="app">
          <Routes>
            <Route path="/login" element={<PublicRoute><LoginPage /></PublicRoute>} />
            <Route path="/register" element={<PublicRoute><RegisterPage /></PublicRoute>} />
            
            <Route path="/" element={
              <ProtectedRoute>
                <>
                  <Navbar />
                  <Navigate to="/dashboard" replace />
                </>
              </ProtectedRoute>
            } />
            
            <Route path="/dashboard" element={
              <ProtectedRoute>
                <>
                  <Navbar />
                  <DashboardPage />
                </>
              </ProtectedRoute>
            } />
            
            <Route path="/schedules" element={
              <ProtectedRoute>
                <>
                  <Navbar />
                  <SchedulePage />
                </>
              </ProtectedRoute>
            } />
            
            <Route path="/settings" element={
              <ProtectedRoute>
                <>
                  <Navbar />
                  <div className="p-8">
                    <h1 className="text-2xl text-white">Settings Page</h1>
                  </div>
                </>
              </ProtectedRoute>
            } />
          </Routes>
        </div>
      </AuthProvider>
    </Router>
  );
};

export default App;