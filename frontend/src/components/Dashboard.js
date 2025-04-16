import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
  CircularProgress,
  AppBar,
  Toolbar,
  IconButton,
} from '@mui/material';
import LogoutIcon from '@mui/icons-material/Logout';
import axios from 'axios';

const Dashboard = () => {
  const [emailContent, setEmailContent] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [user, setUser] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const token = localStorage.getItem('token');
    const userData = localStorage.getItem('user');
    
    if (!token || !userData) {
      navigate('/login');
      return;
    }

    try {
      setUser(JSON.parse(userData));
    } catch (err) {
      console.error('Error parsing user data:', err);
      handleLogout();
    }
  }, [navigate]);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    navigate('/login');
  };

  const handleCheckEmail = async () => {
    if (!emailContent.trim()) {
      setError('Please enter email content');
      return;
    }

    setLoading(true);
    setError('');
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        handleLogout();
        return;
      }

      const response = await axios.post(
        'http://localhost:5000/api/check-email',
        { content: emailContent },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      setResult(response.data);
    } catch (err) {
      if (err.response?.status === 401) {
        handleLogout();
      } else {
        setError(err.response?.data?.message || 'An error occurred while checking the email');
      }
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return null; // or a loading spinner
  }

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      <AppBar position="static" color="primary" elevation={1}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Email Spam Checker
          </Typography>
          <Typography variant="body1" sx={{ mr: 2 }}>
            Welcome, {user.name}
          </Typography>
          <IconButton color="inherit" onClick={handleLogout}>
            <LogoutIcon />
          </IconButton>
        </Toolbar>
      </AppBar>

      <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Paper elevation={3} sx={{ p: 3, borderRadius: 2 }}>
          <Typography variant="h6" gutterBottom sx={{ mb: 2 }}>
            Check Email Content
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={6}
            variant="outlined"
            label="Enter email content"
            value={emailContent}
            onChange={(e) => setEmailContent(e.target.value)}
            sx={{ mb: 2 }}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleCheckEmail}
            disabled={loading}
            sx={{
              py: 1.5,
              borderRadius: 1,
              '&:hover': {
                transform: 'translateY(-2px)',
              },
              transition: 'all 0.3s',
            }}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Check Email'}
          </Button>

          {result && (
            <Alert
              severity={result.is_spam ? 'error' : 'success'}
              sx={{ mt: 2 }}
            >
              {result.is_spam
                ? 'This email is likely SPAM'
                : 'This email is likely NOT SPAM'}
              {result.confidence && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Confidence: {Math.round(result.confidence * 100)}%
                </Typography>
              )}
            </Alert>
          )}
        </Paper>
      </Container>
    </Box>
  );
};

export default Dashboard; 