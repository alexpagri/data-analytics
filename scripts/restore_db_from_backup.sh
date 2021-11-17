#!/usr/bin/env bash

source .env

if [[ -z "$1" ]]; then
  echo "Name of backup file (located in "$DB_BACKUP_LOCAL") required!"
  exit 1
fi

RESTORE_COMMANDS="set "PGPASSWORD=$DB_PASSWORD" && pg_restore -v -c -d "$DB_NAME" "$DB_BACKUP_DOCKER""$1" -U "$DB_USER""

docker exec ma-db-1 bash -c "$RESTORE_COMMANDS"