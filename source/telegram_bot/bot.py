#!/usr/lib/python3.10 python3
# pylint: disable=unused-argument

from dotenv import load_dotenv
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import logging
import datetime

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib
matplotlib.use('Agg')

from source.model.eval import eval_model

import config

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


load_dotenv()

# Define a base command handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"""Hi, {user.mention_html()}! I'am Ploutos bot\n\nI will help you drain your deposit by making shorts on Ethereum =)\n\nAsk me /help to see how to use me""")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Welcome to the Ploutos Bot.\n"
        "There are some commnads:\n"
        "- /send <date> - send command is used to send forecast\n"
        "- /set - set command is used to set schedule of uncoming reports\n"
    )


async def send_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send signals"""

    if not context.args:
        # Get last closed day
        eval_date = pd.Timestamp.today(tz=config.DEFAULT_TZ) - pd.Timedelta(days=1)
    else:
        # Get user arguments
        eval_date = pd.to_datetime(str(context.args[0]), format="%Y-%m-%d", errors="coerce", utc=True)

    if pd.isna(eval_date):
        await update.effective_message.reply_text("Sorry, date needs to be in '%Y-%m-%d' format")
        return

    if eval_date > pd.Timestamp.today(tz=config.DEFAULT_TZ).normalize():
        await update.effective_message.reply_text("Date can't be longer today")
        return

    chat_id = update.effective_message.chat_id

    await context.bot.send_message(chat_id, text=f"Wait signals for 3-5 minute")

    document, symbols = eval_model(eval_date)

    if symbols == []:
        await context.bot.send_message(chat_id, text=f"There aren't signals for {eval_date.strftime('%Y-%m-%d')}")
    else:
        await context.bot.send_message(chat_id, text=f"Beep! New signals for {eval_date.strftime('%Y-%m-%d')} is going")
        await context.bot.send_message(chat_id, text='\n'.join(symbols))
        await context.bot.sendPhoto(chat_id, document)


async def warn_not_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Warn user"""
    await update.message.reply_text("Sorry, but I don't understand you, use /help to see how to use me")


async def error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Printing errors in console"""
    print(f"Update {update} cause error: {context.error}")


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

    await context.bot.send_message(job.chat_id, text=f"Wait signals for 3-5 minute")

    # Get last closed day
    eval_date = pd.Timestamp.today(tz=config.DEFAULT_TZ) - pd.Timedelta(days=1)

    document, symbols = eval_model(valuation_date=eval_date)

    if symbols == []:
        await context.bot.send_message(job.chat_id, text=f"There aren't signals for {eval_date.strftime('%Y-%m-%d')}")
    else:
        await context.bot.send_message(
            job.chat_id, text=f"Beep! New signals for {eval_date.strftime('%Y-%m-%d')} is going"
        )
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
        hours = max(int(context.args[0]) - 5, 0)  # Костыль
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

    token_key= 'TELEGRAM_BOT_TOKEN' if config.MODE == "PROD" else 'TELEGRAM_BOT_TOKEN_TEST'

    BOT_TOKEN = os.getenv(token_key)

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
