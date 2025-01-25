import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LogIn, Lock, Mail } from 'lucide-react';
//import { useAuth } from '../contexts/AuthContext.js';
import { useAuth } from '../../contexts/AuthContext';
const LoginPage = () => {
 const navigate = useNavigate();
 const { login, user } = useAuth();
 const [formData, setFormData] = useState({
   email: '',
   password: ''
 });
 const [error, setError] = useState('');

 useEffect(() => {
   if (user) {
     navigate('/dashboard');
   }
 }, [user, navigate]);

 const handleChange = (e) => {
   setFormData({
     ...formData,
     [e.target.name]: e.target.value
   });
 };

 const handleSubmit = async (e) => {
   e.preventDefault();
   try {
     const result = await login(formData);
     if (!result.success) {
       setError(result.message || 'Login failed');
     }
   } catch (err) {
     setError('Network error occurred');
   }
 };

 return (
   <div className="min-h-screen flex items-center justify-center bg-[#1a1a2e]">
     <div className="w-full max-w-md">
       <div className="bg-[#1e1e2d] p-8 rounded-lg border border-[#2d2d3d] shadow-lg">
         <div className="flex justify-center mb-8">
           <div className="p-3 bg-[#6366f1]/10 rounded-full">
             <LogIn size={32} className="text-[#6366f1]" />
           </div>
         </div>
         
         <h2 className="text-2xl font-bold text-white text-center mb-6">
           Welcome Back
         </h2>

         <form onSubmit={handleSubmit} className="space-y-6">
           <div>
             <label className="flex gap-2 items-center text-gray-300 text-sm mb-2">
               <Mail size={16} />
               Email Address
             </label>
             <input
               type="email"
               name="email"
               value={formData.email}
               onChange={handleChange}
               className="w-full p-3 bg-[#252533] border border-[#2d2d3d] rounded-lg text-white focus:outline-none focus:border-[#6366f1]"
               required
             />
           </div>

           <div>
             <label className="flex gap-2 items-center text-gray-300 text-sm mb-2">
               <Lock size={16} />
               Password
             </label>
             <input
               type="password"
               name="password"
               value={formData.password}
               onChange={handleChange}
               className="w-full p-3 bg-[#252533] border border-[#2d2d3d] rounded-lg text-white focus:outline-none focus:border-[#6366f1]"
               required
             />
           </div>

           {error && (
             <div className="p-3 bg-red-500/20 border border-red-500/50 rounded-lg text-red-400 text-sm">
               {error}
             </div>
           )}

           <button
             type="submit"
             className="w-full py-3 px-4 bg-[#6366f1] hover:bg-[#5355d1] text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2"
           >
             <LogIn size={18} />
             Sign In
           </button>
         </form>
         
         <div className="mt-6 text-center">
           <button 
             onClick={() => navigate('/register')}
             className="text-[#6366f1] hover:underline text-sm"
           >
             Don't have an account? Sign up
           </button>
         </div>
       </div>
     </div>
   </div>
 );
};

export default LoginPage;