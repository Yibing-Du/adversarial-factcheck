# Copyright (c) 2021, Yamagishi Laboratory, National Institute of Informatics
# Author: Canasai Kruengkrai (canasai@nii.ac.jp)
# All rights reserved.
#
# Adapted from https://github.com/facebookresearch/DrQA/blob/master/drqa/retriever/doc_db.py # noqa: E501
# Additional license and copyright information for this source code are available at:
# https://github.com/facebookresearch/DrQA/blob/master/LICENSE

import sqlite3
import unicodedata


class FeverDocDB(object):
    """Sqlite backed document storage."""

    def __init__(self, db_path):
        self.path = db_path
        self.connection = sqlite3.connect(self.path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def get_doc_ids(self):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT id FROM documents")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

    def get_doc_lines(self, doc_id):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        norm_id = unicodedata.normalize("NFD", doc_id)
        cursor.execute(
            "SELECT lines FROM documents WHERE id = ?",
            (norm_id,),
        )
        result = cursor.fetchone()
        cursor.close()
        return result if result is None else result[0]
