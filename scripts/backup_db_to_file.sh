#!/usr/bin/env bash

source .env

if [[ -z "$1" ]]; then
  echo "Name of backup file (without suffix) required!"
  exit 1
fi

BACKUP_COMMANDS="set "PGPASSWORD="$DB_PASSWORD"" && pg_dump -v -U "$DB_USER" -F t "$DB_NAME" > "$DB_BACKUP_DOCKER""$1".tar"

# echo "$BACKUP_COMMANDS"

docker exec ma-db-1 bash -c "$BACKUP_COMMANDS"