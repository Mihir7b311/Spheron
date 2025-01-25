import React, { createContext, useState, useContext, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
 const [user, setUser] = useState(null);
 const [loading, setLoading] = useState(true);
 const navigate = useNavigate();

 const axiosInstance = axios.create({
   baseURL: 'http://localhost:5000/api/auth',
   headers: {
     'Content-Type': 'application/json',
   },
 });

 useEffect(() => {
   checkAuth();
 }, []);

 const checkAuth = async () => {
   const token = localStorage.getItem('token');
   if (!token) {
     setLoading(false);
     return;
   }

   try {
     const response = await axiosInstance.get('/verify', {
       headers: { Authorization: `Bearer ${token}` },
     });

     if (response.status === 200) {
       setUser(response.data.user);
     }
   } catch (err) {
     localStorage.removeItem('token');
   } finally {
     setLoading(false);
   }
 };

 const login = async (credentials) => {
   try {
     const response = await axiosInstance.post('/login', credentials);
     if (response.status === 200) {
       localStorage.setItem('token', response.data.token);
       setUser(response.data.user);
       navigate('/dashboard');
       return { success: true };
     }
   } catch (err) {
     return { 
       success: false, 
       message: err.response?.data?.message || 'Login failed'
     };
   }
 };

 const register = async (userData) => {
   try {
     const response = await axiosInstance.post('/register', userData);
     if (response.status === 200) {
       localStorage.setItem('token', response.data.token);
       setUser(response.data.user);
       navigate('/dashboard');
       return { success: true };
     }
   } catch (err) {
     return {
       success: false,
       message: err.response?.data?.message || 'Registration failed'
     };
   }
 };

 const logout = () => {
   localStorage.removeItem('token');
   setUser(null);
   navigate('/login');
 };

 if (loading) {
   return <div>Loading...</div>;
 }

 return (
   <AuthContext.Provider value={{ user, login, register, logout }}>
     {children}
   </AuthContext.Provider>
 );
};

export const useAuth = () => {
 const context = useContext(AuthContext);
 if (!context) {
   throw new Error('useAuth must be used within an AuthProvider');
 }
 return context;
};