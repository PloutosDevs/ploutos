#!/usr/lib/python3.10 python3
# pylint: disable=unused-argument

import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import logging
import datetime
from io import BytesIO
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

from source import utils
from source.data.get.binance_prices import get_candles_spot_binance
from source.model.eval import eval_model


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# def create_scatter_plot():
#     df = get_candles_spot_binance('BTCUSDT', "1d", "2023-10-01T10:00:00")
#     df = df['Close'][-30:]
#     df.index = df.index.astype(str).str[:10]
#     df.plot()
#     plt.title('Bitcoin Prices (Last 30 Days)')
#     plt.xlabel('Date')
#     plt.ylabel('Price (USD)')
#     # plt.grid(True)
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     # Save the plot as a PNG image
#     img_buffer = BytesIO()
#     plt.savefig(img_buffer, format='png')
#     img_buffer.seek(0)
#     plt.close()

#     return img_buffer

# Define a base command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"""Hi, {user.mention_html()}! I'am Ploutos bot\n\nI will help you drain your deposit by making shorts on Ethereum =)\n\nAsk me /help to see how to use me""")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the Cleint Bot.\n"
        "For this purchase the following commands are available:\n"
        "- /send - send command is to send the log file from the other side of computer"
    )


async def send_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send last 30 d BITCOIN prices plot"""
    chat_id = update.message.chat_id
    document, symbols = eval_model()
    await context.bot.send_message(chat_id, text=f"Beep! New signals is going")
    await context.bot.send_message(chat_id, text='\n'.join(symbols))
    await context.bot.sendPhoto(chat_id, document)


async def warn_not_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Warn user"""
    await update.message.reply_text("Sorry, but I don't understand you, use /help to see how to use me")


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Printing errors in console"""
    await print(f"Update {update} cause error: {context.error}")


async def welcome_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    await context.bot.send_message(chat_id=chat_id, text="Hello! Welcome to the bot.\nTap /start command")
    
########### Daily schedule handlers


async def alarm(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the alarm message."""
    job = context.job
    await context.bot.send_message(job.chat_id, text=f"Beep! {job.data} seconds are over!")


async def send_document_daily(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send last 30 d BITCOIN prices plot"""
    job = context.job
    document, symbols = eval_model()
    await context.bot.send_message(job.chat_id, text=f"Beep! New signals is going")
    await context.bot.send_message(job.chat_id, text='\n'.join(symbols))
    await context.bot.sendPhoto(job.chat_id, document)

def remove_job_if_exists(name: str, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Remove job with given name. Returns whether job was removed."""
    current_jobs = context.job_queue.get_jobs_by_name(name)
    if not current_jobs:
        return False
    for job in current_jobs:
        job.schedule_removal()
    return True


async def set_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add a job to the queue."""
    chat_id = update.effective_message.chat_id
    
    try:
        # args[0] should contain the hour for the Schedule in hours
        hours = max(int(context.args[0]), 0)
        # args[1] should contain the minutes for the Schedule in hours
        minutes = max(int(context.args[1]), 0)
        
        if hours > 23 or hours < 0:
            await update.effective_message.reply_text("Sorry, hour need to be from 0 to 23")
            return
        if minutes > 59 or hours < 0:
            await update.effective_message.reply_text("Sorry, minutes need to be from 0 to 59")
            return
        
        time = datetime.time(hours, minutes)
        
        job_removed = remove_job_if_exists(str(chat_id), context)
        context.job_queue.run_daily(send_document_daily, time=time, days=(0, 1, 2, 3, 4, 5, 6), chat_id=chat_id,
                                    name=str(chat_id), data=time)

        text = f"Schedule successfully set on {time.strftime('%H:%M')} UTC!"
        if job_removed:
            text += "\nOld Schedule was removed."
        await update.effective_message.reply_text(text)

    except (IndexError, ValueError):
        await update.effective_message.reply_text("Usage: /set <hour UTC> <minutes>")


async def unset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove the job if the user changed their mind."""
    chat_id = update.message.chat_id
    job_removed = remove_job_if_exists(str(chat_id), context)
    text = "Schedule successfully cancelled!" if job_removed else "You have no active schedule."
    await update.message.reply_text(text)



def main() -> None:
    """Start the bot."""
    BOT_TOKEN = utils.get_secrets('TELEGRAM_BOT_TOKEN')

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(BOT_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(MessageHandler(filters.StatusUpdate.NEW_CHAT_MEMBERS, welcome_message))
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("send", send_document))
    application.add_error_handler(error)
    
    # Daily sender
    application.add_handler(CommandHandler("set", set_schedule))
    application.add_handler(CommandHandler("unset", unset))

    # on non command i.e message - the message on Telegram
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, warn_not_command))

    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    print("Hello. Cleint has just started.")
    main()
