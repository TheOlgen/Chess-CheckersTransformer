import json
import psycopg2
from dotenv import load_dotenv
import os
class DatabaseManager:

    def __init__(self):
        load_dotenv()
        self.DB_HOST = os.getenv("DB_HOST")
        self.DB_NAME = os.getenv("DB_NAME")
        self.DB_USER = os.getenv("DB_USER")
        self.DB_PASSWORD = os.getenv("DB_PASSWORD")
        self.DB_PORT = os.getenv("DB_PORT")

    def get_connection(self):
        return psycopg2.connect(database=self.DB_NAME, user=self.DB_USER, password=self.DB_PASSWORD, host=self.DB_HOST, port=self.DB_PORT)

    def add_position(self, fen):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('''INSERT INTO public.positions (fen) VALUES (%s)
                                    ON CONFLICT (fen) DO NOTHING
                                ''', (fen,))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_fen_evaluation(self, fen, best_move, score, score_type):

        if (best_move == None or score == None or score_type == None):
            return False

        evaluation = {
            "best_move": best_move,
            "score": score,
            "type": score_type
        }
        conn = self.get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute('''
                                UPDATE public.positions
                                SET eval_json = %s, evaluated = True
                                WHERE fen = %s
                            ''', (json.dumps(evaluation), fen))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_position_to_evaluate(self, limit):
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute('''
                        SELECT fen FROM public.positions WHERE evaluated = False LIMIT %s
                    ''', (limit,))
                return [row[0] for row in cur.fetchall()]
        finally:
            conn.close()