from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from main import TradingBot
import logging
import threading
import time
from datetime import datetime
from logger import log_activity

app = Flask(__name__)

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Add global error handler
@app.errorhandler(Exception)
def handle_error(error):
    logging.error(f"Unhandled error: {str(error)}")
    response = jsonify({
        'status': 'error',
        'message': str(error)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 500

    


# Add new endpoints for activity logs and positions
@app.route('/api/activity-logs', methods=['GET'])
def get_activity_logs():
    try:
        if not check_bot():
            return jsonify({
                'status': 'error',
                'message': 'Trading bot not initialized'
            }), 500
            
        logs = bot.get_activity_logs()
        return jsonify({
            'status': 'success',
            'data': logs
        })
    except Exception as e:
        log_activity(f"Error fetching activity logs: {str(e)}", "error")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/positions', methods=['GET'])
def get_positions():
    try:
        if not check_bot():
            return jsonify({
                'status': 'error',
                'message': 'Trading bot not initialized'
            }), 500
            
        positions = bot.get_positions()
        return jsonify({
            'status': 'success',
            'data': positions
        })
    except Exception as e:
        log_activity(f"Error fetching positions: {str(e)}", "error")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/account-status', methods=['GET'])
def get_account_status():
    try:
        if not check_bot():
            logging.error("Bot check failed in account status endpoint")
            return jsonify({
                'status': 'error',
                'message': 'Trading bot not initialized'
            }), 500
            
        logging.debug("Bot check passed, attempting to fetch account status")
        log_activity("Fetching account status from Alpaca...")
        
        status = bot.get_account_status()
        sentiment_data = bot.get_sentiment_analysis()
        
        if status and sentiment_data:
            status.update(sentiment_data)
        
        logging.debug(f"Returning account status with sentiment: {status}")
        return jsonify({
            'status': 'success',
            'data': status
        })
    except Exception as e:
        logging.error(f"Error in get_account_status: {str(e)}", exc_info=True)
        log_activity(f"Error fetching account status: {str(e)}", "error")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

# Initialize bot globally
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
bot = None
bot_initialized = False

# Make sure all routes are defined before running the app
@app.route('/')
def index():
    try:
        return jsonify({
            'status': 'success',
            'message': 'Trading API is running'
        })
    except Exception as e:
        logging.error(f"Error in index route: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
        
    try:
        if not check_bot():
            return jsonify({
                'status': 'error',
                'message': 'Trading bot not initialized'
            }), 503
            
        return jsonify({
            'status': 'success',
            'data': {
                'bot_initialized': bot_initialized,
                'server_time': datetime.now().isoformat()
            }
        })
    except Exception as e:
        logging.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/test', methods=['GET', 'OPTIONS'])
def test_endpoint():
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
        
    response = jsonify({
        'status': 'success',
        'message': 'API is running',
        'bot_status': {
            'initialized': bot_initialized,
            'exists': bot is not None
        }
    })
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response

@app.route('/debug/bot-status', methods=['GET', 'OPTIONS'])
def debug_bot_status():

    
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
        response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
        return response
        
    response = jsonify({
        'bot_initialized': bot_initialized,
        'bot_exists': bot is not None,
        'symbols': symbols,
        'background_thread_alive': any(t.name == 'bot_background' for t in threading.enumerate())
    })
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    return response

# Add after bot initialization
def check_bot():
    global bot, bot_initialized
    if not bot_initialized:
        try:
            initialize_bot()
        except Exception as e:
            logging.error(f"Failed to initialize bot: {str(e)}")
            return False
    return bot is not None

def initialize_bot():
    global bot, bot_initialized
    if not bot_initialized:
        try:
            log_activity("Starting bot initialization...")
            bot = TradingBot(symbols)
            bot_initialized = True
            log_activity("Trading bot initialized successfully")
            
            # Start the background thread
            thread = threading.Thread(target=run_bot_background, daemon=True)
            thread.name = 'bot_background'
            thread.start()
            log_activity("Background thread started")
            
        except Exception as e:
            log_activity(f"Failed to initialize trading bot: {str(e)}", "error")
            bot_initialized = False
            bot = None
            raise e

def run_bot_background():
    """Run the bot's main loop in the background"""
    while True:
        try:
            if bot and bot.alpaca.get_clock().is_open:
                for symbol in symbols:
                    historical_data = bot.get_historical_data(symbol)
                    if historical_data is not None and bot.feature_columns:
                        latest_data = historical_data.iloc[-1:][bot.feature_columns]
                        prediction, confidence = bot.ensemble_predict(latest_data)
                        if prediction is not None and confidence is not None:
                            bot.execute_trade(symbol, prediction, confidence)
            time.sleep(60)  # Wait 1 minute before next iteration
        except Exception as e:
            logging.error(f"Error in bot background loop: {str(e)}")
            time.sleep(60)

# Add initialization code at the start of api.py
def initialize_application():
    global bot, bot_initialized
    try:
        log_activity("Initializing application...")
        initialize_bot()
        log_activity("Application initialized successfully")
        return True
    except Exception as e:
        log_activity(f"Failed to initialize application: {str(e)}", "error")
        return False

# Update the main execution block
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if initialize_application():
        app.run(
            debug=False,
            host='127.0.0.1',
            port=5000,
            threaded=True,
            use_reloader=False
        )
    else:
        logging.error("Failed to initialize application. Exiting.")

@app.errorhandler(404)
def not_found(error):
    if request.method == 'OPTIONS':
        return build_preflight_response()
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    if request.method == 'OPTIONS':
        return build_preflight_response()
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500
