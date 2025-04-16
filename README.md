# Email Spam Detection System

A robust email spam detection system using BERT and Flask, with a React frontend.

## Features

- Advanced spam detection using BERT model
- User authentication and authorization
- Real-time email content analysis
- Confidence scoring for predictions
- Modern and responsive UI

## Project Structure

```
.
├── frontend/              # React frontend application
│   ├── src/
│   │   ├── components/   # React components
│   │   ├── App.js        # Main application component
│   │   └── index.js      # Entry point
│   └── package.json      # Frontend dependencies
├── backend/              # Flask backend
│   ├── app.py           # Main Flask application
│   ├── predict_spam.py  # Spam detection model
│   └── requirements.txt # Python dependencies
├── saved_model/         # Trained model files
└── README.md           # Project documentation
```

## Setup Instructions

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the Flask server:
```bash
python app.py
```

The backend server will run on http://localhost:5000

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install Node.js dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will run on http://localhost:3000

## API Endpoints

- `POST /api/register` - User registration
- `POST /api/login` - User login
- `POST /api/check-email` - Spam detection (requires authentication)

## Dependencies

### Backend
- Flask 2.3.3
- PyTorch 2.2.1
- Transformers 4.36.2
- NumPy 2.2.4
- scikit-learn 1.6.1
- Other dependencies listed in requirements.txt

### Frontend
- React
- Material-UI
- Axios
- React Router
- Other dependencies listed in package.json

## Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS protection
- Input validation
- Secure session management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 