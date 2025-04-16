import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Button,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  useTheme,
  useMediaQuery,
  alpha,
} from '@mui/material';
import SecurityIcon from '@mui/icons-material/Security';
import EmailIcon from '@mui/icons-material/Email';
import SpeedIcon from '@mui/icons-material/Speed';
import { motion } from 'framer-motion';

const MotionBox = motion.create(Box);
const MotionCard = motion.create(Card);

const LandingPage = () => {
  const navigate = useNavigate();
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  const features = [
    {
      icon: <SecurityIcon sx={{ fontSize: 60, color: 'primary.main' }} />,
      title: 'Advanced Security',
      description: 'Our AI-powered system detects spam with high accuracy using state-of-the-art machine learning algorithms.',
    },
    {
      icon: <EmailIcon sx={{ fontSize: 60, color: 'primary.main' }} />,
      title: 'Real-time Analysis',
      description: 'Get instant results on whether an email is spam or not, with detailed confidence scores.',
    },
    {
      icon: <SpeedIcon sx={{ fontSize: 60, color: 'primary.main' }} />,
      title: 'Lightning Fast',
      description: 'Quick and efficient processing of your emails with minimal waiting time.',
    },
  ];

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
      {/* Hero Section */}
      <Box
        sx={{
          pt: 8,
          pb: 6,
          background: `linear-gradient(135deg, ${theme.palette.primary.main} 0%, ${theme.palette.primary.dark} 100%)`,
          color: 'white',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `radial-gradient(circle at 50% 50%, ${alpha(theme.palette.primary.light, 0.1)} 0%, transparent 50%)`,
          },
        }}
      >
        <Container maxWidth="lg">
          <Grid container spacing={4} alignItems="center">
            <Grid xs={12} md={6}>
              <MotionBox
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.8 }}
              >
                <Typography
                  component="h1"
                  variant={isMobile ? 'h3' : 'h2'}
                  gutterBottom
                  sx={{ 
                    fontWeight: 'bold',
                    textShadow: '2px 2px 4px rgba(0,0,0,0.2)',
                  }}
                >
                  Protect Your Inbox from Spam
                </Typography>
                <Typography 
                  variant="h5" 
                  paragraph 
                  sx={{ 
                    mb: 4,
                    opacity: 0.9,
                  }}
                >
                  Advanced AI-powered spam detection to keep your email clean and secure
                </Typography>
                <Box sx={{ display: 'flex', gap: 2 }}>
                  <Button
                    variant="contained"
                    color="secondary"
                    size="large"
                    onClick={() => navigate('/register')}
                    sx={{ 
                      px: 4, 
                      py: 1.5,
                      borderRadius: 2,
                      boxShadow: 3,
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: 6,
                      },
                      transition: 'all 0.3s',
                    }}
                  >
                    Get Started
                  </Button>
                  <Button
                    variant="outlined"
                    color="inherit"
                    size="large"
                    onClick={() => navigate('/login')}
                    sx={{ 
                      px: 4, 
                      py: 1.5,
                      borderRadius: 2,
                      borderWidth: 2,
                      '&:hover': {
                        borderWidth: 2,
                        transform: 'translateY(-2px)',
                      },
                      transition: 'all 0.3s',
                    }}
                  >
                    Sign In
                  </Button>
                </Box>
              </MotionBox>
            </Grid>
            <Grid xs={12} md={6}>
              <MotionBox
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.8, delay: 0.2 }}
              >
                <Paper
                  elevation={6}
                  sx={{
                    p: 4,
                    bgcolor: 'rgba(255, 255, 255, 0.1)',
                    backdropFilter: 'blur(10px)',
                    borderRadius: 3,
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                  }}
                >
                  <Typography variant="h5" gutterBottom sx={{ fontWeight: 'bold' }}>
                    Try it now!
                  </Typography>
                  <Typography variant="body1" paragraph sx={{ opacity: 0.9 }}>
                    Enter any email content to check if it's spam
                  </Typography>
                  <Button
                    variant="contained"
                    color="secondary"
                    onClick={() => navigate('/register')}
                    fullWidth
                    sx={{
                      py: 1.5,
                      borderRadius: 2,
                      boxShadow: 3,
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: 6,
                      },
                      transition: 'all 0.3s',
                    }}
                  >
                    Check Email
                  </Button>
                </Paper>
              </MotionBox>
            </Grid>
          </Grid>
        </Container>
      </Box>

      {/* Features Section */}
      <Container maxWidth="lg" sx={{ py: 8 }}>
        <MotionBox
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <Typography
            variant="h3"
            align="center"
            gutterBottom
            sx={{ 
              mb: 6, 
              fontWeight: 'bold',
              color: 'primary.main',
            }}
          >
            Why Choose Us?
          </Typography>
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid xs={12} md={4} key={index}>
                <MotionCard
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: index * 0.2 }}
                  sx={{
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    borderRadius: 3,
                    overflow: 'hidden',
                    boxShadow: 3,
                    '&:hover': {
                      transform: 'translateY(-8px)',
                      boxShadow: 6,
                    },
                    transition: 'all 0.3s',
                  }}
                >
                  <CardContent sx={{ 
                    flexGrow: 1, 
                    textAlign: 'center',
                    p: 4,
                  }}>
                    {feature.icon}
                    <Typography 
                      variant="h5" 
                      component="h2" 
                      gutterBottom 
                      sx={{ 
                        mt: 2,
                        fontWeight: 'bold',
                        color: 'primary.main',
                      }}
                    >
                      {feature.title}
                    </Typography>
                    <Typography 
                      variant="body1" 
                      color="text.secondary"
                      sx={{ 
                        lineHeight: 1.6,
                      }}
                    >
                      {feature.description}
                    </Typography>
                  </CardContent>
                </MotionCard>
              </Grid>
            ))}
          </Grid>
        </MotionBox>
      </Container>

      {/* CTA Section */}
      <Box
        sx={{
          py: 8,
          background: `linear-gradient(135deg, ${theme.palette.secondary.main} 0%, ${theme.palette.secondary.dark} 100%)`,
          color: 'white',
          position: 'relative',
          overflow: 'hidden',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `radial-gradient(circle at 50% 50%, ${alpha(theme.palette.secondary.light, 0.1)} 0%, transparent 50%)`,
          },
        }}
      >
        <Container maxWidth="md">
          <MotionBox
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Typography 
              variant="h3" 
              align="center" 
              gutterBottom
              sx={{ 
                fontWeight: 'bold',
                textShadow: '2px 2px 4px rgba(0,0,0,0.2)',
              }}
            >
              Ready to Protect Your Inbox?
            </Typography>
            <Typography 
              variant="h6" 
              align="center" 
              paragraph
              sx={{ 
                opacity: 0.9,
                mb: 4,
              }}
            >
              Join thousands of users who trust our spam detection system
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'center' }}>
              <Button
                variant="contained"
                color="primary"
                size="large"
                onClick={() => navigate('/register')}
                sx={{ 
                  px: 6, 
                  py: 1.5,
                  borderRadius: 2,
                  boxShadow: 3,
                  '&:hover': {
                    transform: 'translateY(-2px)',
                    boxShadow: 6,
                  },
                  transition: 'all 0.3s',
                }}
              >
                Start Free Trial
              </Button>
            </Box>
          </MotionBox>
        </Container>
      </Box>
    </Box>
  );
};

export default LandingPage; 