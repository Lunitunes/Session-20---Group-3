'use client'

import { useState } from "react"
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button"
import { Controller, useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { z } from "zod"
import { FieldGroup, Field, FieldLabel, FieldError,  } from "@/components/ui/field";
import { uploadSchema } from "@/schema/uploadSchema";


export default function InputComponent(){
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState("");

  const form = useForm<z.infer<typeof uploadSchema>>({
    defaultValues: {
      analysisName:""
    },
    resolver: zodResolver(uploadSchema)
  })

  function onSubmit() {
    // Chuck backend stuff here
  }

  return(
    <div className="grid max-w-sm items-center m-auto w-5xl">
      <h1 className="text-3xl my-5 text-center">Input a CSV to get started</h1>
      <div className="grid gap-2 shadow-lg p-4 rounded-lg bg-card w-full border-2">
        <form onSubmit={form.handleSubmit(onSubmit)}>
          <FieldGroup>
            <Field>
              <FieldLabel htmlFor="analysisName">Analysis Name:</FieldLabel>
              <Input id="analyisName" type="text"/>
              <FieldError />
            </Field>
            <Field>
              <FieldLabel htmlFor="file">Upload File:</FieldLabel>
              <Input id="file" type="file"/>
              <FieldError />
            </Field>
          </FieldGroup>

        <Button type="submit" className="w-full"/>
        </form>
      </div>
    </div>
  )
}